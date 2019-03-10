package com.example.vijay.sentimentanalysis_ondevice;


import android.content.Context;
import android.os.Environment;
import android.util.Log;
import android.util.Pair;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class SentimentClassifier {

    public Context context;
    private static String TAG = "SentimentClassifier";
    private static String MODEL_PATH = Environment.getExternalStorageDirectory().toString()+"/SENTIMENT_DATA/";
    private static String TRAINING_DATA_PATH = Environment.getExternalStorageDirectory().toString()+"/SENTIMENT_DATA/training/";
    private static String TESTING_DATA_PATH = Environment.getExternalStorageDirectory().toString()+"/SENTIMENT_DATA/testing/";
    private WordVectorSaver wordVectorSaver;
    private WordVectorReader wordVectorReader;
    private final List<String> stopwords = new ArrayList<String>();
    private final List<String> extendedStopwords = new ArrayList<String>();
    private static String ModelFile = "Sentiment.text";
    private static File modelsavedfile;
    boolean status =false;

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    public SentimentClassifier(Context context) throws IOException {

        this.context = context;

        wordVectorSaver = new WordVectorSaver(context);
        wordVectorReader = new WordVectorReader(context);

        /*Load Stopwords*/
        InputStream stop = context.getResources().openRawResource(R.raw.stopwords);
        InputStream exstop = context.getResources().openRawResource(R.raw.extended_stopwords);

        BufferedReader br = new BufferedReader(new InputStreamReader(stop));
        String line;
        while((line = br.readLine()) != null){
            stopwords.add(line);
        }
        br.close();
        br = new BufferedReader(new InputStreamReader(exstop));
        while((line = br.readLine()) != null){
            extendedStopwords.add(line);
        }
        br.close();

        File datapathFiles = new File(MODEL_PATH);
        if(!datapathFiles.exists()){
            datapathFiles.mkdir();
        }
        modelsavedfile = new File(MODEL_PATH+File.separator+ModelFile);
        if(!modelsavedfile.exists()){
            modelsavedfile.createNewFile();
            status = true; /*Training required for with sentiment data*/
        }

        if(status){
            wordVectorSaver.resetSharedpreferences();
        }else{
            wordVectorSaver.setSharedpreferences();
        }
    }

    public void SentimentClassifier(){}

    public void  makeParagraphVectors()  throws Exception {

        if(wordVectorSaver.getSavedModelState() == false) {

            File datapathFiles = new File(TRAINING_DATA_PATH);
            if (!datapathFiles.exists()) {
                Log.e(TAG, "Missing testing datapath");
            }
            // build a iterator for our dataset
            iterator = new FileLabelAwareIterator.Builder()
                    .addSourceFolder(datapathFiles)
                    .build();

            tokenizerFactory = new DefaultTokenizerFactory();
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

            // ParagraphVectors training configuration
            paragraphVectors = new ParagraphVectors.Builder()
                    .learningRate(0.025)
                    .minLearningRate(0.001)
                    .batchSize(1000)
                    .epochs(20)
                    .iterate(iterator)
                    .trainWordVectors(true)
                    .stopWords(stopwords)
                    .stopWords(extendedStopwords)
                    .tokenizerFactory(tokenizerFactory)
                    .build();

            // Start model training
            paragraphVectors.fit();

            wordVectorSaver.writeParagraphVectors(paragraphVectors,modelsavedfile);
            wordVectorSaver.setSharedpreferences();
        }else{
            paragraphVectors = wordVectorReader.readParagraphVectors(modelsavedfile);
        }
    }

    Collection<String> checkUnlabeledData(String rawText){
        Collection<String >list;
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);
        INDArray textAsCentroid = meansBuilder.textAsVector(rawText);
        list = paragraphVectors.nearestLabels(textAsCentroid,1); /*Nearest top first labels*/
        return list;
    }


    public void checkUnlabeledData() throws IOException {
      /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */
      /* Keep label data for testing [POS/pos.txt or NEG/neg.txt] */

        File datapathFiles = new File(TESTING_DATA_PATH);
        if(!datapathFiles.exists()){
            Log.e(TAG,"Missing testing datapath");
        }

        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(datapathFiles)
                .build();

     /*
      Now we'll iterate over unlabeled data, and check which label it could be assigned to
      Please note: for many domains it's normal to have 1 document fall into few labels at once,
      with different "weight" for each.
     */
        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
                tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument document = unClassifiedIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

         /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
         */
            Log.i(TAG,"Document '" + document.getLabels() + "' falls into the following categories: ");
            for (Pair<String, Double> score: scores) {
                Log.i(TAG,"        " + score.first + ": " + score.second);
            }
        }
    }



}
