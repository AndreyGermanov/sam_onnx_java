package org.sam;

import ai.onnxruntime.*;
import io.javalin.Javalin;
import io.javalin.http.Context;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

public class Main {

    public static OnnxTensor image_embeddings = null;
    public static float orig_width = 0;
    public static float orig_height = 0;
    public static float resized_width = 0;
    public static float resized_height = 0;
    public static OrtEnvironment env = null;
    public static OrtSession encoder = null;
    public static OrtSession decoder = null;

    public static void main(String[] args) {
        env = OrtEnvironment.getEnvironment();
        try {
            encoder = env.createSession("vit_t_encoder.onnx", new OrtSession.SessionOptions());
            decoder = env.createSession("vit_t_decoder.onnx", new OrtSession.SessionOptions());
            Javalin.create().get("/", ctx -> ctx.html(Files.readString(Path.of("index.html"))))
                .post("/encode", Main::encode)
                .post("/decode", Main::decode)
                .start(8080);
        } catch (OrtException e) {
            System.out.println("Could not load model: '"+e.getMessage()+"'");
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    static void encode(Context ctx) {
        try {
            BufferedImage img = ImageIO.read(ctx.uploadedFile("image_file").content());
            float[][][][] input_tensor = prepare_input(img);
            run_encoder(input_tensor);
            ctx.result("OK");
        } catch (OrtException e) {
            System.out.println("Could not create a session: '"+e.getMessage()+"'");
            ctx.status(500);
        } catch (IOException e) {
            System.out.println("Could not read an image: '"+e.getMessage()+"'");
            ctx.status(500);
        }
    }

    static float[][][][] prepare_input(BufferedImage img) {
        orig_width = img.getWidth();
        orig_height = img.getHeight();
        if (orig_width > orig_height) {
            resized_width = 1024;
            resized_height = 1024 / orig_width * orig_height;
        } else {
            resized_height = 1024;
            resized_width = 1024 / orig_height * orig_width;
        }

        BufferedImage resized_img = new BufferedImage((int)resized_width, (int)resized_height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = resized_img.createGraphics();
        graphics2D.drawImage(img, 0, 0, (int)resized_width, (int)resized_height, null);

        float[] mean = {123.675f, 116.28f, 103.53f};
        float[] std = {58.395f, 57.12f, 57.375f};
        float [][][][] input_tensor = new float[1][3][1024][1024];

        for (int y=0;y<(int)resized_height;y++) {
            for (int x=0;x<(int)resized_width;x++) {
                int color = resized_img.getRGB(x, y);
                input_tensor[0][0][y][x] = ( (float)((color & 0xff0000) >> 16) - mean[0] ) / std[0];
                input_tensor[0][1][y][x] = ( (float)((color & 0xff0000) >> 8) - mean[1] ) / std[1];
                input_tensor[0][2][y][x] = ( (float)(color & 0xff) - mean[2] ) / std[2];
            }
        }
        return input_tensor;
    }

    static void run_encoder(float[][][][] input_tensor) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession encoder = env.createSession("vit_t_encoder.onnx", new OrtSession.SessionOptions());
        Map <String,OnnxTensor> inputs = new HashMap<>();
        inputs.put("images",OnnxTensor.createTensor(env,input_tensor));
        OrtSession.Result results = encoder.run(inputs);
        image_embeddings = (OnnxTensor)results.get("embeddings").get();
    }

    static void decode(Context ctx) {
        String boxStr = ctx.formParam("box");
        String[] items = boxStr.split(",");
        int x1 = Integer.parseInt(items[0]);
        int y1 = Integer.parseInt(items[1]);
        int x2 = Integer.parseInt(items[2]);
        int y2 = Integer.parseInt(items[3]);
        float[][][] point_coords = prepare_prompt(x1, y1, x2, y2);
        try {
            ctx.json(decode_mask(point_coords, x1, y1, x2, y2));
        } catch (OrtException e) {
            System.out.println(e.getMessage());
            ctx.status(500);
        }
    }

    static float[][][] prepare_prompt(int x1,int y1,int x2,int y2) {
        float coefX = resized_width / orig_width;
        float coefY = resized_height / orig_height;
        return new float[][][]{{{x1*coefX,y1*coefY},{x2*coefX,y2*coefY}}};
    };

    static float[] decode_mask(float[][][] point_coords, int x1, int y1, int x2, int y2) throws OrtException {
        float[][] point_labels = {{2.0f,3.0f}};
        float[] orig_im_size = {orig_height,orig_width};
        float[][][][] mask_input = new float[1][1][256][256];
        float[] has_mask_input = {0.0f};
        Map <String,OnnxTensor> inputs = new HashMap<>();
        inputs.put("image_embeddings", image_embeddings);
        inputs.put("point_coords", OnnxTensor.createTensor(env, point_coords));
        inputs.put("point_labels", OnnxTensor.createTensor(env, point_labels));
        inputs.put("orig_im_size", OnnxTensor.createTensor(env, orig_im_size));
        inputs.put("mask_input", OnnxTensor.createTensor(env, mask_input));
        inputs.put("has_mask_input", OnnxTensor.createTensor(env, has_mask_input));
        OrtSession.Result results = decoder.run(inputs);
        OnnxTensor output = (OnnxTensor)results.get("masks").get();
        float[] mask = output.getFloatBuffer().array();
        float[] result = new float[(y2-y1)*(x2-x1)];
        int index = 0;
        for (int y=y1; y<y2; y++) {
            for (int x=x1; x<x2; x++) {
                result[index] = mask[y*(int)orig_width+x] < 0 ? 0 : 1;
                index++;
            }
        }
        return result;
    }

}