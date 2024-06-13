package app.ij.mlwithtensorflowlite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Model;

public class MainActivity extends AppCompatActivity {

    // Deklarasi variabel untuk UI
    TextView result, benefits, sideEffects;
    ImageView imageView;
    Button picture;
    int imageSize = 224;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Inisialisasi variabel UI
        result = findViewById(R.id.result);
        benefits = findViewById(R.id.benefits); // Ditambahkan untuk menampilkan manfaat
        sideEffects = findViewById(R.id.sideEffects); // Ditambahkan untuk menampilkan efek samping
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        // Listener untuk tombol gambar
        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Meluncurkan kamera jika izin telah diberikan
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    // Meminta izin kamera jika belum diberikan
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    // Metode untuk mengklasifikasikan gambar
    public void classifyImage(Bitmap image){
        try {
            // Memuat model TensorFlow Lite
            Model model = Model.newInstance(getApplicationContext());

            // Membuat input untuk referensi
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Mendapatkan array 1D dari 224 * 224 piksel pada gambar
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // Iterasi melalui piksel dan ekstrak nilai R, G, dan B. Menambahkannya ke bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Menjalankan inferensi model dan mendapatkan hasil
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // Menemukan indeks kelas dengan kepercayaan terbesar
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"APEL", "NANAS", "PEPAYA", "JAMBU", "JERUK"};
            String[] benefitsArray = {
                    "Menjaga kesehatan pencernaan, menurunkan risiko penyakit jantung, meningkatkan kesehatan otak, dan mengurangi risiko kanker.",
                    "Meningkatkan daya tahan tubuh, menjaga kesehatan tulang, membantu penyembuhan luka pasca operasi, dan menurunkan asam urat.",
                    "Melancarkan pencernaan, meningkatkan kesehatan kulit, mengurangi peradangan, dan meningkatkan fungsi ginjal.",
                    "Mengatur tekanan darah, menyehatkan kulit, membantu mengendalikan gula darah, dan menyegarkan napas.",
                    "Meningkatkan sistem imun, menurunkan kolesterol, mengurangi risiko batu ginjal, dan mendukung kesehatan kulit."
            };
            String[] sideEffectsArray = {
                    "Menyebabkan berat badan sulit turun, merusak gigi, dan membuat gula darah naik turun.",
                    "Membuat perut kembung, memicu gejala maag, dan menyebabkan lidah gatal dan panas.",
                    "Dapat merusak kerongkongan, menimbulkan efek pencahar seperti sakit perut dan diare, serta dapat memicu perut kembung.",
                    "Dapat mengganggu kinerja obat diabetes pada penderita diabetes dan potensi menyebabkan radang usus buntu.",
                    "Memperburuk gejala penyakit GERD serta dapat menyebabkan masalah pencernaan dan memicu reaksi alergi pada beberapa orang."
            };

            // Menampilkan hasil, manfaat, dan efek samping pada UI
            result.setText(classes[maxPos]);
            benefits.setText(benefitsArray[maxPos]); // Menampilkan teks manfaat
            sideEffects.setText(sideEffectsArray[maxPos]); // Menampilkan teks efek samping

            // Menutup model untuk membebaskan sumber daya
            model.close();
        } catch (IOException e) {
            // TODO: Tangani pengecualian
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            // Mendapatkan gambar dari kamera
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            // Mengubah ukuran gambar ke 224x224 piksel
            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            // Memanggil metode classifyImage untuk mengklasifikasikan gambar
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
