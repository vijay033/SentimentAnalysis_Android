package com.example.vijay.sentimentanalysis_ondevice;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;

public class MainActivity extends AppCompatActivity {

    TextView sentimentText, sentimentLabel;
    ProgressBar prgbar;
    Context context;
    private SentimentClassifier sentimentClassifier;
    Collection<String> sentimentLabels;
    String senText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sentimentText = (TextView)findViewById(R.id.sentimentText);
        sentimentLabel = (TextView)findViewById(R.id.sentimentLabel);
        prgbar = (ProgressBar)findViewById(R.id.progressBar);
        prgbar.setVisibility(View.INVISIBLE);

        context = this.getApplicationContext();

        senText = "Worthless to watch it"; /*Negative*/
//        senText = "Worth to watch it"; /*Positive*/

        sentimentText.setText(senText);

        try {
            sentimentClassifier = new SentimentClassifier(context);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Runnable runnable = new Runnable() {
            @Override
            public void run() {

                prgbar.setVisibility(View.VISIBLE);

                try {
                    sentimentClassifier.makeParagraphVectors();
                    prgbar.post(new Runnable() {
                        @Override
                        public void run() {
                            prgbar.setVisibility(View.INVISIBLE);
//                            try {
//                                sentimentClassifier.checkUnlabeledData();
//                            } catch (IOException e) {
//                                e.printStackTrace();
//                            }

                            String text ="";
                            sentimentLabels = sentimentClassifier.checkUnlabeledData(senText);
                            Iterator<String> itr = sentimentLabels.iterator();
                            while(itr.hasNext()){
                                text += itr.next();
                                text += "";
                            }
                            sentimentLabel.setText(text);
                        }
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        };
        new Thread(runnable).start();

    }
}
