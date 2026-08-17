import io.scif.config.SCIFIOConfig;
import io.scif.services.DatasetIOService;
import mpicbg.ij.InteractiveInvertibleCoordinateTransform;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imglib2.*;
import net.imglib2.algorithm.fft2.FFTConvolution;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.ImgView;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.realtransform.*;
import net.imglib2.realtransform.interval.IntervalSamplingMethod;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.ValuePair;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;
import org.apache.commons.lang3.tuple.MutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.function.Computers;
import org.scijava.log.LogService;
import org.scijava.ops.api.OpBuilder;
import org.scijava.ops.api.OpEnvironment;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.types.Nil;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.scijava.ItemVisibility.MESSAGE;

@Plugin(type = Command.class, headless = true, menuPath = "Process>Registration>Batch 3D Registration")
public class Batch_3D_Registration <T extends RealType<T>, R extends InvertibleRealTransform> extends DynamicCommand {

    @Parameter
    protected LogService logService;

    @Parameter
    protected StatusService statusService;

    @Parameter
    protected UIService uiService;

    @Parameter
    protected DatasetService datasetService;

    @Parameter
    protected DatasetIOService datasetIOService;

    @Parameter
    protected OpEnvironment ops;

    @Parameter
    protected OpService ijOps;

    @Parameter (visibility = MESSAGE, required=false)
    protected String msg = "Set a reference image that all moving images will align to:";

    @Parameter(label = "Reference image: ")
    protected File referenceFile;

    @Parameter(label = "Moving images: ")
    protected File[] movingFiles;

    @Parameter(label = "Threshold (Raw pixel value): ", required = false)
    protected Integer thresholdValue;

    @Parameter(label = "Registered images output directory: ", style = "directory")
    protected File saveFolder;

    protected SCIFIOConfig config;

//    protected Interval workingInterval;
    protected AffineTransform3D originOffset;
    protected CorrelationTranslation translator;
    protected Rotation rotator;

    protected RealTransformRandomAccessible offsetReferenceView;


    protected void errorChecking(Dataset referenceImage){
        if(referenceImage.numDimensions() != 3){
            throw new RuntimeException(new IOException("This plugin does not currently support images that are not exactly 3 dimensions."));
        }
    }

    protected void initializePlugin(Dataset referenceImage){
        if(saveFolder != null) saveFolder.mkdirs();

        config = new SCIFIOConfig();
        config.writerSetFailIfOverwriting(false);

        translator = new CorrelationTranslation();

        originOffset = new AffineTransform3D();

        originOffset.set( 1,0,0,-(referenceImage.dimension(0)/2.0),
                0,1,0,-(referenceImage.dimension(1)/2.0),
                0,0,1,-(referenceImage.dimension(2)/2.0));

//        workingInterval = new FinalInterval(new long[]{(long)-Math.floor(referenceImage.dimension(0)/2.0), (long)-Math.floor(referenceImage.dimension(1)/2.0), (long)-Math.floor(referenceImage.dimension(0)/2.0)},
//                new long[]{(long)Math.ceil(referenceImage.dimension(0)/2.0), (long)Math.ceil(referenceImage.dimension(1)/2.0), (long)Math.ceil(referenceImage.dimension(0)/2.0)});

        //Since we don't have to save the reference, we don't need to create a Thresholded copy
        applyThreshold((RandomAccessibleInterval<T>) referenceImage, thresholdValue);

        RealRandomAccessible interpolatedReference = extendZeroAndInterpolate(referenceImage);

        offsetReferenceView = RealViews.transform(interpolatedReference, originOffset);

        rotator = new Rotation(referenceImage);

    }

    @Override
    public void run(){
        logService.info("Starting Batch 3D Registration plugin");
        Dataset referenceImage;

        if(!datasetIOService.canOpen(referenceFile.getPath())){
            throw new RuntimeException(new IOException("Cannot open reference file as image."));
        }
        logService.info("Opening reference image file: " + referenceFile.getPath());
        try {
            referenceImage = datasetIOService.open(referenceFile.getPath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        errorChecking(referenceImage);
        logService.info("Initializing plugin");
        initializePlugin(referenceImage);

        for (File movingFile:movingFiles){

            //Set up the running combined transform
            AffineTransform3D currentTransform = new AffineTransform3D();
            currentTransform.set(originOffset);

            if(!datasetIOService.canOpen(movingFile.getPath())){
                logService.warn("Skipping non-openable moving file: " + movingFile.getPath());
                continue;
            }
            logService.info("Opening moving image file: " + movingFile.getPath());
            Dataset originalMovingImage;
            try {
                originalMovingImage = datasetIOService.open(movingFile.getPath());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            //Need to create a copy, so we can save the original image.
            logService.info("Applying threshold to moving image");
            Img<T> movingImage = (Img<T>) originalMovingImage.copy();

            applyThreshold(movingImage,thresholdValue);

            //define cube up here?

            RealRandomAccessible<T> interpolatedMoving = extendZeroAndInterpolate(movingImage);
            Interval centeredMovingCurrentInterval = new FinalInterval(movingImage);

            //Need to somehow translate the data, and then make that the center of the interval...
            //for loop for sampling iterations starts here; do a test run with up to 5 loops at FULL sample rate, and save images at each iteration

            for (int i = 0; i < 1; ++i) {

                RealTransformRandomAccessible<? extends RealType, ? extends InvertibleRealTransform> currentMovingView = RealViews.transform(interpolatedMoving, currentTransform);

                //calculate  and appplyTranslation
                currentTransform.preConcatenate(
                        translationViaCenterOfGravity(
                                getCurrentWorkingIntervalView(offsetReferenceView, referenceImage),
                                getCurrentWorkingIntervalView(currentMovingView, movingImage)
                        )
                );


                //currentTransform.translate(translator.findMovingImageTranslation(referenceImage, getCurrentZeroMinWorkingIntervalView(currentMovingView, originalMovingImage)));
                currentMovingView = RealViews.transform(interpolatedMoving, currentTransform);
//                uiService.show("Centered reference", Views.zeroMin(Views.interval(offsetReferenceView, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
//                uiService.show("Centered moving",Views.zeroMin(Views.interval(currentMovingView, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));


                //showStackedImages("Post-translation" + i ,getCurrentWorkingIntervalView(offsetReferenceView, referenceImage), getCurrentWorkingIntervalView(currentMovingView, referenceImage));

                currentTransform.preConcatenate(
                        rotator.findFullImageRotation(
                                getCurrentWorkingIntervalView(offsetReferenceView, referenceImage),
                                getCurrentWorkingIntervalView(currentMovingView, movingImage)
                        )
                );
                currentMovingView = RealViews.transform(interpolatedMoving, currentTransform);

                showStackedImages("Post rotation " + i, getCenteredImage(getCurrentWorkingIntervalView(offsetReferenceView, referenceImage)), getCenteredImage(getCurrentWorkingIntervalView(currentMovingView, movingImage)));

//                uiService.show(referenceImage.getName(), getCurrentZeroMinWorkingIntervalView(offsetReferenceView, referenceImage));
//                uiService.show(originalMovingImage.getName(), getCurrentZeroMinWorkingIntervalView(currentMovingView, referenceImage));

            }

            movingImage = null;

            Dataset output = datasetService.create(
                getCurrentZeroMinWorkingIntervalView(
                    RealViews.transform(
                        extendZeroAndInterpolate(originalMovingImage),
                        currentTransform),
                    referenceImage
                )
            );

            try {
                datasetIOService.save(output, saveFolder.getPath() + File.separator + "Registered-" + originalMovingImage.getName() + ".tif", config);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }
    }

    private RandomAccessibleInterval getCenteredImage(RandomAccessibleInterval input){
        return Views.zeroMin(Views.interval(input, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200})));
    }

    private void showStackedImages(String name, RandomAccessibleInterval<? extends RealType> input1, RandomAccessibleInterval<? extends RealType> input2){
        Interval combined = Intervals.union(input1,input2);
        uiService.show(name, datasetService.create(
                ImgView.wrap(Views.zeroMin(
                        Views.stack(
                                Views.interval(Views.extendZero(input1), combined),
                                Views.interval(Views.extendZero(input2), combined)
                        )
                ))
                )
        );
    }

    private void applyThreshold(RandomAccessibleInterval<T> input, Integer thresholdValue){
        LoopBuilder.setImages(input).multiThreaded().forEachPixel((p) ->{
            if(p.getRealDouble() < thresholdValue){
                p.setReal(0.0);
            }
        });
    }

    private Interval getCurrentWorkingInterval(InvertibleRealTransform transform, Interval originalInterval){
        return Intervals.smallestContainingInterval(
                transform.inverse().boundingInterval(originalInterval, IntervalSamplingMethod.CORNERS)
        );
    }

    //Can add scale factor to this? Do a Scale transform of the view before rastering
    private RandomAccessibleInterval getCurrentWorkingIntervalView(RealTransformRealRandomAccessible<? extends RealType, ? extends InvertibleRealTransform> view, Interval originalInterval){
        return Views.interval(
                Views.raster(
                        view
                ),
                getCurrentWorkingInterval(view.getTransformToSource(), originalInterval)
        );
    }


    private RandomAccessibleInterval getCurrentZeroMinWorkingIntervalView(RealTransformRealRandomAccessible<? extends RealType, ? extends InvertibleRealTransform> view, Interval interval){
        return Views.interval(
                    Views.raster(
                        RealViews.transform(view, originOffset.inverse())
                    ),
                interval
        );
    }

    private RealRandomAccessible extendZeroAndInterpolate(RandomAccessibleInterval input){
        return Views.interpolate(
                (ExtendedRandomAccessibleInterval)ops.op("transform.extendZeroView").input(input).apply(),
                new NLinearInterpolatorFactory()
        );
    }


    public AffineTransform3D translationViaCenterOfGravity(RandomAccessibleInterval reference, RandomAccessibleInterval moving){

        AffineTransform3D translationAffine = new AffineTransform3D();
        translationAffine.identity();

        ExecutorService executor = Executors.newFixedThreadPool(2);

        RealPoint[] referenceCenter = new RealPoint[1];
        RealPoint[] movingCenter = new RealPoint[1];

        CompletableFuture<Void> future1 = CompletableFuture.runAsync(() -> {
            referenceCenter[0] = ((RealPoint) ops.op("geom.centerOfGravity").input(reference).apply());
        }, executor);

        CompletableFuture<Void> future2 = CompletableFuture.runAsync(() -> {
            // Run second op simultaneously
            movingCenter[0] = ((RealPoint) ops.op("geom.centerOfGravity").input(moving).apply());
        }, executor);

        CompletableFuture.allOf(future1,future2).join();

        double[] translation = new double[3];

        for (int i = 0; i <3; i++) {
            translation[i] = referenceCenter[0].getDoublePosition(i)-movingCenter[0].getDoublePosition(i);
        }
        logService.info("Translating moving image by: " + translation[0] + "," +translation[1] + "," +translation[2]);
        translationAffine.translate(translation);
        return translationAffine;
    }

    public class Rotation{

        Interval sphericalInterval;
        long rotationalSampleRate;

        AffineTransform3D rotate0to2;
        AffineTransform3D rotate1to2;

        public Rotation(Interval input){

            rotate0to2 = new AffineTransform3D();
            rotate1to2 = new AffineTransform3D();

            rotate0to2.set( 0,0,1,0,
                    0,1,0,0,
                    -1,0,0,0);

            rotate1to2.set( 1,0,0,0,
                    0,0,-1,0,
                    0,1,0,0);

            rotationalSampleRate = 10;

            double maxDist = 0.0;
            long[] dims = input.dimensionsAsLongArray();
            for(long dim:dims){
                maxDist += Math.pow(dim/2.0, 2);
            }
            //spherical coordinates are dist, inclination, azimuth
            maxDist = Math.round(Math.sqrt(maxDist)*1.1);
            maxDist += maxDist%2;
            //I think the range here may be off, for the inclination and azimuthal axes
            long inclinationRange = getScaledSphericalHalfDim(rotationalSampleRate);
            inclinationRange += inclinationRange%2;
            sphericalInterval = new FinalInterval(new long[]{0, 0, -getScaledSphericalHalfDim(rotationalSampleRate)},new long[]{(long)maxDist, inclinationRange, getScaledSphericalHalfDim(rotationalSampleRate)});
        }

        private long getScaledSphericalHalfDim(long scale){
            long value = Math.round((Math.PI*scale));
            return value;
        }


        //class for Rotation via flat Sum intensity projection
        public AffineTransform3D findFullImageRotation(RandomAccessibleInterval<? extends RealType> reference, RandomAccessibleInterval<? extends RealType> moving){
            AffineTransform3D rotation = new AffineTransform3D();
            rotation.identity();
            HashMap<Integer,Img<FloatType>> axesToSearch = new HashMap<>(reference.numDimensions());

//            uiService.show("Moving in findFullImageRotation", datasetService.create(ImgView.wrap(moving)));


//            showStackedImages("FullImageRotationInput", reference, moving);

            axesToSearch.put(0, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
            axesToSearch.put(1, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
            axesToSearch.put(2, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());

            RealRandomAccessible interpolatedMovingView = extendZeroAndInterpolate(moving);

            while (!axesToSearch.isEmpty()){
                RealTransformRealRandomAccessible currentView = RealViews.transformReal(interpolatedMovingView, rotation);

                AffineTransform3D currentRotation = new AffineTransform3D();
                currentRotation.identity();

                ValuePair<Integer,Double> toRotate = findHighestRotationCorrelation(reference,getCurrentWorkingIntervalView(currentView, moving), axesToSearch);
                System.out.println("Rotating image about axis " + toRotate.getA() + " by " + toRotate.getB()+ " radians.");
                axesToSearch.remove(toRotate.getA());
                currentRotation.rotate(toRotate.getA(), toRotate.getB());
                rotation.preConcatenate(currentRotation);
            }

//            uiService.show("Centered moving in Rotation",Views.zeroMin(Views.interval(RealViews.transform(interpolatedMovingView, rotation), new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
//
//            showStackedImages("FullImageRotationBeforeReturn", reference, getCurrentWorkingIntervalView(RealViews.transformReal(interpolatedMovingView, rotation), moving));
            return rotation;
        }

        public <T extends RealType<T>> ValuePair<Integer, Double> findHighestRotationCorrelation(RandomAccessibleInterval<T> reference, RandomAccessibleInterval<T> moving, HashMap<Integer,Img<FloatType>> toSearch){
//            uiService.show("Moving in findHighestRotationCorrelation", datasetService.create(ImgView.wrap(moving)));

            for (Integer axis: toSearch.keySet()){
                RandomAccessibleInterval rotatedReferenceView = null;
                RandomAccessibleInterval rotatedMovingView = null;

                switch (axis){
                    case 0:
                        rotatedReferenceView = rotateImage(reference, rotate0to2);
                        rotatedMovingView = rotateImage(moving, rotate0to2);
                        break;
                    case 1:
                        rotatedReferenceView = rotateImage(reference, rotate1to2);
                        rotatedMovingView = rotateImage(moving, rotate1to2);
                        break;
                    case 2:
                        rotatedReferenceView = reference;
                        rotatedMovingView = moving;
                        break;
                }

                setSingleAxisCorrelationImage(toSearch.get(axis), rotatedReferenceView, rotatedMovingView);

                uiService.show(axis + "-axis corr", Views.zeroMin(Views.rotate(toSearch.get(axis), 0, 2)));
            }
            return findHighestRotationalPointAmong(toSearch);
        }

        private RandomAccessibleInterval rotateImage(RandomAccessibleInterval input, AffineTransform3D transform){
            return getCurrentWorkingIntervalView(
                    RealViews.transform(extendZeroAndInterpolate(input), transform),
                    input
            );
        }

        private void setSingleAxisCorrelationImage(Img<FloatType> correlationImage, RandomAccessibleInterval<? extends RealType> reference, RandomAccessibleInterval<? extends RealType> moving){
            FFTConvolution fftConvolution;
            ExecutorService service = Executors.newCachedThreadPool();

//            uiService.show("Reference spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(reference)), 0, 2));
//            uiService.show("Moving spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(moving)), 0, 2));

            fftConvolution = new FFTConvolution(Views.extendPeriodic(convertToSphericalCoordinates(reference)), correlationImage,Views.extendPeriodic(convertToSphericalCoordinates(moving)), correlationImage, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(correlationImage, new ComplexFloatType()).apply(), service);
            fftConvolution.setComputeComplexConjugate(true);
            fftConvolution.setOutput(correlationImage);
            fftConvolution.convolve();
        }


        private ValuePair<Integer, Double> findHighestRotationalPointAmong(HashMap<Integer,Img<FloatType>> toSearch){
            //returns dimension and azimuthal angle for rotation
            float max = 0.0F;
            ValuePair<Integer,Long> maxPoint;
            maxPoint = new ValuePair<>(-1, -1L);

            for (Integer i: toSearch.keySet()) {
                //todo: Finish this and make sure the dimesion is right
                RandomAccessibleInterval<FloatType> rotationalCorrelation =
                        Views.hyperSlice(
                                Views.hyperSlice(toSearch.get(i), 0, (toSearch.get(i).dimension(0)-1)/2),
                                0, (toSearch.get(i).dimension(1)-1)/2);
                Cursor<FloatType> cursor = rotationalCorrelation.localizingCursor();

                while (cursor.hasNext()){
                    if(cursor.next().get() > max){
                        max = cursor.get().get();
                        maxPoint = new ValuePair<>(i,cursor.getLongPosition(0));
                    }
                }
                System.out.println(maxPoint.getA() + "," + maxPoint.getB());
            }
            double fractionalAngle = (maxPoint.getB() - ((toSearch.get(maxPoint.getA()).dimension(2)-1)/2.0))/(toSearch.get(maxPoint.getA()).dimension(2)-1);

            if(maxPoint.getA() == -1){
                return new ValuePair<>(0,0.0);
            }
            return new ValuePair<>(maxPoint.getA(), 2*Math.PI*fractionalAngle);
        }

        private RandomAccessibleInterval convertToSphericalCoordinates(RandomAccessibleInterval input){

            RealTransformRealRandomAccessible sphericalInput = RealViews.transform(
                    RealViews.transformReal(extendZeroAndInterpolate(input), SphericalToCartesianTransform3D.getInstance().inverse()),
                    new Scale3D(1, rotationalSampleRate, rotationalSampleRate)
            );

            return Views.interval(
                    Views.raster(
                            sphericalInput
                    ),
                    sphericalInterval
            );
        }

//        private RandomAccessibleInterval convertToLogPolarCoordinates(RandomAccessibleInterval input, Interval region){
//            ScaledPolarToTranslatedCartesianTransform2D polarTransform = new ScaledPolarToTranslatedCartesianTransform2D(0, 0, 1, 360/(2*Math.PI));
//
//            //todo: Need to do log polar transform?
//            RealTransformRealRandomAccessible polarInput = RealViews.transformReal(extendZeroAndInterpolate(input), polarTransform.inverse());
//
//
//
//            return Views.interval(
//                    Views.raster(
//                            polarInput
//                    ),
//                    polarInterval
//            );
//        }

    }

    public class CorrelationTranslation {

        public double[] findMovingImageTranslation(RandomAccessibleInterval reference, RandomAccessibleInterval moving){
            CorrelationFunctions correlationFunctions = new CorrelationFunctions();
            Img<FloatType> correlationImage = correlationFunctions.correlate(reference,moving);

            double[] center = getCenter(correlationImage);
            double[] max = findMaxPoint(correlationImage);

            double[] translation = new double[center.length];

            for (int i = 0; i < center.length; i++) {
                translation[i] = max[i]-center[i];
            }
            return translation;
        }

        private double[] getCenter(Interval interval){
            double[] center = new double[interval.numDimensions()];
            for (int i = 0; i < center.length; i++) {
                center[i] = ((interval.max(i)-interval.min(i))/2.0);
            }
            return center;
        }

        private double[] findMaxPoint(IterableInterval<FloatType> image) {
            float max = 0;
            double[] maxPoint = new double[image.numDimensions()];
            Cursor<FloatType> cursor = image.localizingCursor();

            while (cursor.hasNext()){
                if(cursor.next().get() > max){
                    max = cursor.get().get();
                    maxPoint = cursor.positionAsDoubleArray();
                }
            }
            return maxPoint;
        }
    }

    public class CorrelationFunctions {

//        public RandomAccessibleInterval<FloatType> correlate(RandomAccessibleInterval input, RandomAccessibleInterval kernel){
//            Interval bounds = new FinalInterval(greatestDimensions(input, kernel));
//
//            return (RandomAccessibleInterval<FloatType>) ops.op("filter.correlate").input(
//                    expandImage(input, bounds),
//                    expandImage(kernel, bounds),
//                    new FloatType(),
//                    new ComplexFloatType(),
//                    null,
//                    null,
//                    null)
//                    .apply();
//        }

        public Img<FloatType> correlate(RandomAccessibleInterval input, RandomAccessibleInterval kernel){
            FFTConvolution fftConvolution;
            ExecutorService service = Executors.newCachedThreadPool();
            Dimensions dims = greatestOddDimensions(input, kernel);

            Img<FloatType> crossCorrelation = (Img<FloatType>) ops.op("create.img").input(dims, new FloatType()).apply();

            Interval bounds = new FinalInterval(crossCorrelation);

            fftConvolution = new FFTConvolution(Views.extendZero(input), bounds,Views.extendZero(kernel), bounds, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(bounds, new ComplexFloatType()).apply(), service);
            fftConvolution.setComputeComplexConjugate(true);
            fftConvolution.setOutput(crossCorrelation);
            fftConvolution.convolve();

            return crossCorrelation;
        }

        private Dimensions greatestOddDimensions(RealInterval input1, RealInterval input2){
            long[] dim1 = Intervals.smallestContainingInterval(input1).dimensionsAsLongArray();
            long[] dim2 = Intervals.smallestContainingInterval(input2).dimensionsAsLongArray();

            //Odd dimensions results give an integer image center, which gives better translation results, so we make them odd

            long[] output = new long[dim1.length];
            for (int i = 0; i < output.length; i++) {
                output[i] = Math.max(dim1[i], dim2[i]);
                if(output[i]%2==0)
                    output[i]++;
            }

            return new FinalDimensions(output);
        }

    }
}
