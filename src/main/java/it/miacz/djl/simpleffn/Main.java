package it.miacz.djl.simpleffn;


import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {


    public static void main(String[] args) throws IOException, TranslateException {
        int inputSize = 28*28;
        int outputSize = 10;
        //region INFO: (GM) These are just some layers. They are not used in this example
        SequentialBlock sequentialBlock = new SequentialBlock();
        sequentialBlock.add(Blocks.batchFlattenBlock(inputSize));
        sequentialBlock.add(Linear.builder().setUnits(128).build());
        sequentialBlock.add(Activation::relu);
        sequentialBlock.add(Linear.builder().setUnits(64).build());
        sequentialBlock.add(Activation::relu);
        sequentialBlock.add(Linear.builder().setUnits(outputSize).build());
        //endregion
        int batchSize = 32;
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());

        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(inputSize, outputSize, new int[]{128,64}));
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(1, inputSize));

        int epoch = 2;

        EasyTrain.fit(trainer, epoch, mnist, null);

        Path modelDir = Paths.get("build/mlp");
        Files.createDirectories(modelDir);

        model.setProperty("Epoch", String.valueOf(epoch));

        model.save(modelDir, "mlp");
        System.out.println(model);

        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");
        img.getWrappedImage();


        Translator<Image, Classifications> translator = new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                return new NDList(NDImageUtils.toTensor(array));
            }

            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                NDArray probabilities = list.singletonOrThrow().softmax(0);
                List<String> classNames = IntStream.range(0, outputSize).mapToObj(String::valueOf).collect(Collectors.toList());
                return new Classifications(classNames, probabilities);
            }

            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }
        };

        var predictor = model.newPredictor(translator);

        var classifications = predictor.predict(img);

        System.out.println(classifications);
    }
}
