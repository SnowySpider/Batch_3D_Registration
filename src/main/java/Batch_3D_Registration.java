import com.google.common.collect.EvictingQueue;
import io.scif.config.SCIFIOConfig;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
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
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.ValuePair;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;
import org.apache.commons.lang3.tuple.MutableTriple;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.ops.api.OpEnvironment;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.scijava.ItemVisibility.MESSAGE;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>Registration>Batch 3D Registration")
public class Batch_3D_Registration <T extends RealType<T>, R extends InvertibleRealTransform> implements Command {

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

    @Parameter(label = "Registration accuracy: ")
    protected B3dParameters.Resolutions resolutionChoice = B3dParameters.Resolutions.Normal;

    @Parameter(label = "Moving images: ")
    protected File[] movingFiles;

    @Parameter(label = "Threshold (Raw pixel value): ", required = false)
    protected Double thresholdValue;

    @Parameter(label = "Registered images output directory: ", style = "directory")
    protected File saveFolder;

    protected SCIFIOConfig config;

//    protected Interval workingInterval;
    protected CorrelationTranslation translator;
    protected Rotation2D rotator;
    protected AffineTransform3D originOffset;
    protected RealTransformRandomAccessible offsetReferenceView;
    protected LinkedList<B3dParameters> resolutionList;

    //protected RealTransformRandomAccessible offsetReferenceView;


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

        //Since we don't have to save the reference, we don't need to create a Thresholded copy
        applyThreshold((RandomAccessibleInterval<T>) referenceImage, thresholdValue);

        originOffset = new AffineTransform3D();
        originOffset.set( 1,0,0,-(referenceImage.dimension(0)/2.0),
                0,1,0,-(referenceImage.dimension(1)/2.0),
                0,0,1,-(referenceImage.dimension(2)/2.0));
        offsetReferenceView = RealViews.transform(extendZeroAndInterpolate(referenceImage), originOffset);

        resolutionList = new LinkedList<>();

        switch (resolutionChoice){
            case Native:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Native));
            case VeryHigh:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.VeryHigh));
            case High:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.High));
            case Normal:
            default:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Normal));
            case Moderate:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Moderate));
            case Low:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Low));
            case VeryLow:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.VeryLow));
        }
        resolutionList.addFirst(new B3dParameters(referenceImage, 50000, 15.0, false));
        resolutionList.addFirst(new B3dParameters(referenceImage, 25000, 20.0, false));
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
            RealRandomAccessible extendedInterpolatedMoving = extendZeroAndInterpolate(movingImage);

            //Set up the running combined transform
            AffineTransform3D currentTransform = new AffineTransform3D();
            currentTransform.set(originOffset);

            for(B3dParameters resolution:resolutionList) {

                logService.info("Starting alignment at " + resolution.getSubsampleScale()[0] + " scaling factor, and "
                + (180/(resolution.getRotationalScaleFactor()*Math.PI)) + " degrees rotational accuracy.");

                RealTransformRandomAccessible currentMovingView = RealViews.transform(extendedInterpolatedMoving, currentTransform);

                RandomAccessibleInterval subSampledReference = getSubsampledWorkingIntervalView(
                        offsetReferenceView,
                        referenceImage,
                        resolution.getSubsampleScale()
                );

                RandomAccessibleInterval subSampledMoving = getSubsampledWorkingIntervalView(
                        currentMovingView,
                        movingImage,
                        resolution.getSubsampleScale()
                );

//                showStackedImages("subSampled Centered", getCenteredImage(subSampledReference), getCenteredImage(subSampledMoving));

                //still need to offset original reference image?
                currentTransform.preConcatenate(
                        scaleTransform(
                                alignSubsampledImages(
                                        subSampledReference,
                                        subSampledMoving,
                                        resolution
                                ),
                                resolution.getSubsampleScale()
                        )
                );
            }

//            currentTransform.concatenate(originOffset);

            Dataset output = datasetService.create(
                    ImgPlus.wrap(ImgView.wrap(getCurrentWorkingIntervalView(
                        RealViews.transform(
                            RealViews.transform(
                                extendZeroAndInterpolate(originalMovingImage),
                                currentTransform
                            ),
                            originOffset.inverse()
                        ),
                        referenceImage
                    )),
                    originalMovingImage.getImgPlus())
            );

            logService.info("Saving registered image:" + movingFile.getPath());
            try {
                datasetIOService.save(output, saveFolder.getPath() + File.separator + "Registered-" + originalMovingImage.getName() + ".tif", config);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        logService.info("Batch 3D registration finished.");
    }

    private AffineTransform3D scaleTransform(AffineTransform3D input, double[] scale){
        input.set(input.get(0,3)/scale[0],0,3);
        input.set(input.get(1,3)/scale[1],1,3);
        input.set(input.get(2,3)/scale[2],2,3);

        return input;
    }


    private AffineTransform3D alignSubsampledImages(RandomAccessibleInterval referenceSubsampled, RandomAccessibleInterval movingSubsampled, B3dParameters resolutionParams){

//        showStackedImages("Passed in", getCenteredImage(referenceSubsampled), getCenteredImage(movingSubsampled));
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

//        AffineTransform3D subsampledRefOffset = new AffineTransform3D();
//
//        subsampledRefOffset.set( 1,0,0,-(referenceSubsampled.dimension(0)/2.0),
//                0,1,0,-(referenceSubsampled.dimension(1)/2.0),
//                0,0,1,-(referenceSubsampled.dimension(2)/2.0));

        rotator = new Rotation2D(referenceSubsampled, resolutionParams.getRotationalScaleFactor());
        RealRandomAccessible interpolatedReference = extendZeroAndInterpolate(referenceSubsampled);

        AffineTransform3D subsampledAlignment = new AffineTransform3D();
        subsampledAlignment.identity();

        RealRandomAccessible<T> interpolatedMoving = extendZeroAndInterpolate(movingSubsampled);

        //Need to somehow translate the data, and then make that the center of the interval...
        //for loop for sampling iterations starts here; do a test run with up to 5 loops at FULL sample rate, and save images at each iteration

        Double transformationMagnitude = Double.POSITIVE_INFINITY;
        int i = 0;
        Double transformationMagnitudeTarget = 0.1;
        EvictingQueue<Double> lastThreeMagnitudes = EvictingQueue.create(4);

        //may need to check for a lack of change in transformation magnitude
        while(transformationMagnitude > transformationMagnitudeTarget){



            double[] initialTransformation = subsampledAlignment.getRowPackedCopy();

            RealTransformRandomAccessible<? extends RealType, ? extends InvertibleRealTransform> subsampledMovingView = RealViews.transform(interpolatedMoving, subsampledAlignment);

            //calculate  and applyTranslation
            if(resolutionParams.getTranslateByCorrelate()){
                logService.info("Calculating translation via cross-correlation.");
                subsampledAlignment.translate(
                        translator.findMovingImageTranslation(
                                referenceSubsampled,
                                getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)
                        )
                );
            }
            else {
                logService.info("Calculating translation via center of mass.");
                subsampledAlignment.preConcatenate(
                        translationViaCenterOfGravity(
                                referenceSubsampled,
                                getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)
                        )
                );
            }


            //currentTransform.translate(translator.findMovingImageTranslation(referenceImage, getCurrentZeroMinWorkingIntervalView(currentMovingView, originalMovingImage)));
            subsampledMovingView = RealViews.transform(interpolatedMoving, subsampledAlignment);
//                uiService.show("Centered reference", Views.zeroMin(Views.interval(offsetReferenceView, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
//                uiService.show("Centered moving",Views.zeroMin(Views.interval(currentMovingView, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));


//            showStackedImages("Post translation " + i, getCenteredImage(referenceSubsampled), getCenteredImage(getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)));

            subsampledAlignment.preConcatenate(
                    rotator.findFullImageRotation(
                            referenceSubsampled,
                            getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)
                    )
            );
            subsampledMovingView = RealViews.transform(interpolatedMoving, subsampledAlignment);

            transformationMagnitude = getTransformationMagnitude(initialTransformation,subsampledAlignment.getRowPackedCopy());

            lastThreeMagnitudes.add(transformationMagnitude);

            if(isCircular(lastThreeMagnitudes)){
                int randAxis = (int) Math.round(Math.random()*3);
                double randRotation = Math.random()*(2*Math.PI)-Math.PI;
                logService.info("Transformations are suspected to be circular, rotating randomly about axis "
                        + randAxis + " by " +Math.toDegrees(randRotation) + " degrees.");
                subsampledAlignment.rotate(randAxis,randRotation);
            }

            if(i > 10){
                transformationMagnitudeTarget += transformationMagnitude/10;
            }

            logService.info("Total transformation magnitude change during iteration " + i + ": " + transformationMagnitude);

//            showStackedImages("Post rotation " + i, getCenteredImage(referenceSubsampled), getCenteredImage(getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)));

//                uiService.show(referenceImage.getName(), getCurrentZeroMinWorkingIntervalView(offsetReferenceView, referenceImage));
//                uiService.show(originalMovingImage.getName(), getCurrentZeroMinWorkingIntervalView(currentMovingView, referenceImage));
            i++;
        }

        return subsampledAlignment;
    }

    private double getTransformationMagnitude(double[] initial, double[] post){
        double magnitude = 0.0;
        for (int j = 0; j < initial.length; j++) {
            magnitude += Math.pow((post[j]-initial[j]),2);
        }
        return Math.sqrt(magnitude);
    }

    private boolean isCircular(EvictingQueue<Double> input){
        if(input.remainingCapacity() > 0)
            return false;
        Double initialValue = input.peek();
        Double totalDiff = 0.0;
        for(Double value: input){
            totalDiff += Math.abs(value-initialValue);
        }
        return totalDiff < 0.01;
    }

    private RandomAccessibleInterval getCenteredImage(RandomAccessibleInterval input){

        return Views.zeroMin(
                Views.interval(
                        input,
                        new FinalInterval(new long[]{-20, -20, -20}, new long[]{20,20,20})
                )
        );
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

    private void applyThreshold(RandomAccessibleInterval<T> input, Double thresholdValue){
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


    private RandomAccessibleInterval getSubsampledWorkingIntervalView(RealTransformRealRandomAccessible<? extends RealType, ? extends InvertibleRealTransform> view, Interval originalInterval, double[] subsampleFactor){
        Scale3D scale3D = new Scale3D(subsampleFactor);

        Interval preScaleInterval = getCurrentWorkingInterval(view.getTransformToSource(), originalInterval);
        long[] min = preScaleInterval.minAsLongArray();
        long[] max = preScaleInterval.maxAsLongArray();

        for (int i = 0; i < min.length; i++) {
            min[i] = (long)Math.floor(min[i]*subsampleFactor[i]);
            max[i] = (long)Math.ceil(max[i]*subsampleFactor[i]);
        }

        //Scaling is working, interval calculation is not.
        return Views.interval(
                    Views.raster(
                        RealViews.transform(view, scale3D)
                    ),
                new FinalInterval(min, max)
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
        logService.info("Translating scaled moving image by: " + translation[0] + "," +translation[1] + "," +translation[2]);
        translationAffine.translate(translation);
        return translationAffine;
    }

//    public class Rotation{
//
//        Interval sphericalInterval;
//        double rotationalScaleFactor;
//
//        AffineTransform3D rotate0to2;
//        AffineTransform3D rotate1to2;
//
//        public Rotation(Interval input, double rotationalScaleFactor){
//
//            rotate0to2 = new AffineTransform3D();
//            rotate1to2 = new AffineTransform3D();
//
//            rotate0to2.set( 0,0,1,0,
//                    0,1,0,0,
//                    -1,0,0,0);
//
//            rotate1to2.set( 1,0,0,0,
//                    0,0,-1,0,
//                    0,1,0,0);
//
//            this.rotationalScaleFactor = rotationalScaleFactor;
//
//            double maxDist = 0.0;
//            long[] dims = input.dimensionsAsLongArray();
//            for(long dim:dims){
//                maxDist += Math.pow(dim/2.0, 2);
//            }
//            //spherical coordinates are dist, inclination, azimuth
//            maxDist = Math.round(Math.sqrt(maxDist)*1.1);
//            maxDist += maxDist%2;
//            //I think the range here may be off, for the inclination and azimuthal axes
//            long inclinationRange = getScaledSphericalHalfDim(rotationalScaleFactor);
//            inclinationRange += inclinationRange%2;
//            sphericalInterval = new FinalInterval(new long[]{0, 0, -getScaledSphericalHalfDim(rotationalScaleFactor)},new long[]{(long)maxDist, inclinationRange, getScaledSphericalHalfDim(rotationalScaleFactor)});
//        }
//
//        private long getScaledSphericalHalfDim(double scale){
//            long value = Math.round((Math.PI*scale));
//            return value;
//        }
//
//
//        //class for Rotation via flat Sum intensity projection
//        public AffineTransform3D findFullImageRotation(RandomAccessibleInterval<? extends RealType> reference, RandomAccessibleInterval<? extends RealType> moving){
//            AffineTransform3D rotation = new AffineTransform3D();
//            rotation.identity();
//            HashMap<Integer,Img<FloatType>> axesToSearch = new HashMap<>(reference.numDimensions());
//
////            uiService.show("Moving in findFullImageRotation", datasetService.create(ImgView.wrap(moving)));
//
//
////            showStackedImages("FullImageRotationInput", reference, moving);
//
//            axesToSearch.put(0, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
//            axesToSearch.put(1, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
//            axesToSearch.put(2, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
//
//            RealRandomAccessible interpolatedMovingView = extendZeroAndInterpolate(moving);
//
//            while (axesToSearch.size()>1){
//                //This rotates two axes at most before re-translating and starting again.
//                RealTransformRealRandomAccessible currentView = RealViews.transformReal(interpolatedMovingView, rotation);
//
//                AffineTransform3D currentRotation = new AffineTransform3D();
//                currentRotation.identity();
//
//                ValuePair<Integer,Double> toRotate = findHighestRotationCorrelation(reference,getCurrentWorkingIntervalView(currentView, moving), axesToSearch);
//                logService.info("Rotating image about axis " + toRotate.getA() + " by " + Math.toDegrees(toRotate.getB())+ " degrees.");
//                axesToSearch.remove(toRotate.getA());
//                currentRotation.rotate(toRotate.getA(), toRotate.getB());
//                rotation.preConcatenate(currentRotation);
//            }
//
////            uiService.show("Centered moving in Rotation",Views.zeroMin(Views.interval(RealViews.transform(interpolatedMovingView, rotation), new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
////
////            showStackedImages("FullImageRotationBeforeReturn", reference, getCurrentWorkingIntervalView(RealViews.transformReal(interpolatedMovingView, rotation), moving));
//            return rotation;
//        }
//
//        public <T extends RealType<T>> ValuePair<Integer, Double> findHighestRotationCorrelation(RandomAccessibleInterval<T> reference, RandomAccessibleInterval<T> moving, HashMap<Integer,Img<FloatType>> toSearch){
////            uiService.show("Moving in findHighestRotationCorrelation", datasetService.create(ImgView.wrap(moving)));
//
//            for (Integer axis: toSearch.keySet()){
//                RandomAccessibleInterval rotatedReferenceView = null;
//                RandomAccessibleInterval rotatedMovingView = null;
//
//                switch (axis){
//                    case 0:
//                        rotatedReferenceView = rotateImage(reference, rotate0to2);
//                        rotatedMovingView = rotateImage(moving, rotate0to2);
//                        break;
//                    case 1:
//                        rotatedReferenceView = rotateImage(reference, rotate1to2);
//                        rotatedMovingView = rotateImage(moving, rotate1to2);
//                        break;
//                    case 2:
//                        rotatedReferenceView = reference;
//                        rotatedMovingView = moving;
//                        break;
//                }
//
//                setSingleAxisCorrelationImage(toSearch.get(axis), rotatedReferenceView, rotatedMovingView);
//
////                uiService.show(axis + "-axis corr", Views.zeroMin(Views.dropSingletonDimensions(Views.hyperSlice(toSearch.get(axis), 0, (toSearch.get(axis).dimension(0)-1)/2))));
//            }
//            return findHighestRotationalPointAmong(toSearch);
//        }
//
//        private RandomAccessibleInterval rotateImage(RandomAccessibleInterval input, AffineTransform3D transform){
//            return getCurrentWorkingIntervalView(
//                    RealViews.transform(extendZeroAndInterpolate(input), transform),
//                    input
//            );
//        }
//
//        private void setSingleAxisCorrelationImage(Img<FloatType> correlationImage, RandomAccessibleInterval<? extends RealType> reference, RandomAccessibleInterval<? extends RealType> moving){
//            FFTConvolution fftConvolution;
//            ExecutorService service = Executors.newCachedThreadPool();
//
////            uiService.show("Reference spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(reference)), 0, 2));
////            uiService.show("Moving spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(moving)), 0, 2));
//
//            fftConvolution = new FFTConvolution(Views.extendPeriodic(convertToSphericalCoordinates(reference)), correlationImage,Views.extendPeriodic(convertToSphericalCoordinates(moving)), correlationImage, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(correlationImage, new ComplexFloatType()).apply(), service);
//            fftConvolution.setComputeComplexConjugate(true);
//            fftConvolution.setOutput(correlationImage);
//            fftConvolution.convolve();
//        }
//
//
//        private ValuePair<Integer, Double> findHighestRotationalPointAmong(HashMap<Integer,Img<FloatType>> toSearch){
//            //returns dimension and azimuthal angle for rotation
//            float max = 0.0F;
//            ValuePair<Integer,Long> maxPoint = new ValuePair<>(-1, -1L);
//
//
//            boolean repeatSearch = true;
//
//            while(repeatSearch && !toSearch.isEmpty()) {
//                max = 0.0F;
//                maxPoint = new ValuePair<>(-1, -1L);
//
//                for (Integer i : toSearch.keySet()) {
//
//                    RandomAccessibleInterval<FloatType> rotationalCorrelation =
//                            Views.hyperSlice(
//                                    Views.hyperSlice(toSearch.get(i), 0, (toSearch.get(i).dimension(0) - 1) / 2),
//                                    0, (toSearch.get(i).dimension(1) - 1) / 2);
//                    Cursor<FloatType> cursor = rotationalCorrelation.localizingCursor();
//
//                    while (cursor.hasNext()) {
//                        if (cursor.next().get() > max) {
//                            max = cursor.get().get();
//                            maxPoint = new ValuePair<>(i, cursor.getLongPosition(0));
//                        }
//                    }
//                }
//
//                if(maxPoint.getB() == ((toSearch.get(maxPoint.getA()).dimension(2)-1)/2.0) ) {
//                    //Rotation of 0 result
//                    toSearch.remove(maxPoint.getA());
//                }
//                else{
//                    repeatSearch = false;
//                }
//            }
//            if(maxPoint.getA() == -1 || toSearch.isEmpty()){
//                return new ValuePair<>(0,0.0);
//            }
//
//            double fractionalAngle = (maxPoint.getB() - ((toSearch.get(maxPoint.getA()).dimension(2)-1)/2.0))/(toSearch.get(maxPoint.getA()).dimension(2)-1);
//
//            return new ValuePair<>(maxPoint.getA(), 2*Math.PI*fractionalAngle);
//        }
//
//        private RandomAccessibleInterval convertToSphericalCoordinates(RandomAccessibleInterval input){
//
//            RealTransformRealRandomAccessible sphericalInput = RealViews.transform(
//                    RealViews.transformReal(extendZeroAndInterpolate(input), SphericalToCartesianTransform3D.getInstance().inverse()),
//                    new Scale3D(1, rotationalScaleFactor, rotationalScaleFactor)
//            );
//
//            return Views.interval(
//                    Views.raster(
//                            sphericalInput
//                    ),
//                    sphericalInterval
//            );
//        }
//
////        private RandomAccessibleInterval convertToLogPolarCoordinates(RandomAccessibleInterval input, Interval region){
////            ScaledPolarToTranslatedCartesianTransform2D polarTransform = new ScaledPolarToTranslatedCartesianTransform2D(0, 0, 1, 360/(2*Math.PI));
////
////            //todo: Need to do log polar transform?
////            RealTransformRealRandomAccessible polarInput = RealViews.transformReal(extendZeroAndInterpolate(input), polarTransform.inverse());
////
////
////
////            return Views.interval(
////                    Views.raster(
////                            polarInput
////                    ),
////                    polarInterval
////            );
////        }
//
//    }

    public class Rotation2D {

        Interval sphericalInterval;
        double rotationalScaleFactor;

        AffineTransform3D rotate0to2;
        AffineTransform3D rotate1to2;

        public Rotation2D(Interval input, double rotationalScaleFactor) {

            rotate0to2 = new AffineTransform3D();
            rotate1to2 = new AffineTransform3D();

            rotate0to2.set(0, 0, 1, 0,
                    0, 1, 0, 0,
                    -1, 0, 0, 0);

            rotate1to2.set(1, 0, 0, 0,
                    0, 0, -1, 0,
                    0, 1, 0, 0);

            this.rotationalScaleFactor = rotationalScaleFactor;

            double maxDist = 0.0;
            long[] dims = input.dimensionsAsLongArray();
            for (long dim : dims) {
                maxDist += Math.pow(dim / 2.0, 2);
            }
            //spherical coordinates are dist, inclination, azimuth
            maxDist = Math.round(Math.sqrt(maxDist) * 1.1);
            maxDist += maxDist % 2;
            //I think the range here may be off, for the inclination and azimuthal axes
            long inclinationRange = getScaledSphericalHalfDim(rotationalScaleFactor);
            inclinationRange += inclinationRange % 2;
            sphericalInterval = new FinalInterval(new long[]{0, 0, -getScaledSphericalHalfDim(rotationalScaleFactor)}, new long[]{(long) maxDist, inclinationRange, getScaledSphericalHalfDim(rotationalScaleFactor)});
        }

        private long getScaledSphericalHalfDim(double scale) {
            long value = Math.round((Math.PI * scale));
            return value;
        }


        //class for Rotation via flat Sum intensity projection
        public AffineTransform3D findFullImageRotation(RandomAccessibleInterval<T> reference, RandomAccessibleInterval<? extends RealType> moving) {
            HashMap<Integer, Img<FloatType>> axesToSearch = new HashMap<>(reference.numDimensions());

            AffineTransform3D currentRotation = new AffineTransform3D();
            currentRotation.identity();
//            uiService.show("Moving in findFullImageRotation", datasetService.create(ImgView.wrap(moving)));


//            showStackedImages("FullImageRotationInput", reference, moving);

            axesToSearch.put(0, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
            axesToSearch.put(1, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
            axesToSearch.put(2, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());

            RealRandomAccessible interpolatedMovingView = extendZeroAndInterpolate(moving);

            RealTransformRealRandomAccessible currentView = RealViews.transformReal(interpolatedMovingView, currentRotation);

            MutableTriple<Integer, Double,Double> possibleRotations = findHighestRotationCorrelation(reference, getCurrentWorkingIntervalView(currentView, moving), axesToSearch);

            List<ValuePair<Integer,Double>> bestOrderedRotations = getOptimalRotationOrder(reference, currentView, possibleRotations);

            logService.info("Rotating image about axis " + bestOrderedRotations.get(0).getA() + " by " + Math.toDegrees(bestOrderedRotations.get(0).getB()) + " degrees, then about "
            + bestOrderedRotations.get(1).getA()+ " by " + Math.toDegrees(bestOrderedRotations.get(1).getB()));

            currentRotation.rotate(bestOrderedRotations.get(0).getA(), bestOrderedRotations.get(0).getB());
            currentRotation.rotate(bestOrderedRotations.get(1).getA(), bestOrderedRotations.get(1).getB());


//            uiService.show("Centered moving in Rotation",Views.zeroMin(Views.interval(RealViews.transform(interpolatedMovingView, rotation), new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
//
//            showStackedImages("FullImageRotationBeforeReturn", reference, getCurrentWorkingIntervalView(RealViews.transformReal(interpolatedMovingView, rotation), moving));
            return currentRotation;
        }

        public <T extends RealType<T>> MutableTriple<Integer, Double, Double> findHighestRotationCorrelation(RandomAccessibleInterval<T> reference, RandomAccessibleInterval<T> moving, HashMap<Integer, Img<FloatType>> toSearch) {
//            uiService.show("Moving in findHighestRotationCorrelation", datasetService.create(ImgView.wrap(moving)));

            for (Integer axis : toSearch.keySet()) {
                RandomAccessibleInterval rotatedReferenceView = null;
                RandomAccessibleInterval rotatedMovingView = null;

                switch (axis) {
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

//                uiService.show(axis + "-axis corr", Views.zeroMin(Views.dropSingletonDimensions(Views.hyperSlice(toSearch.get(axis), 0, (toSearch.get(axis).dimension(0) - 1) / 2))));
            }

            return findHighestRotationalPointAmong(toSearch);
        }


        //This does not seem to work????
        public List<ValuePair<Integer,Double>> getOptimalRotationOrder(RandomAccessibleInterval<T> reference, RealTransformRealRandomAccessible<T, AffineTransform3D> moving, MutableTriple<Integer, Double, Double> rotations){
            if(Math.abs(rotations.getMiddle()) < Double.MIN_VALUE){
                List<ValuePair<Integer,Double>> returnList = new ArrayList<ValuePair<Integer,Double>>(2);
                returnList.add(new ValuePair<>(rotations.getLeft(), rotations.getRight()));
                returnList.add(new ValuePair<>(0, 0.0));

                return returnList;
            }

            List<List<ValuePair<Integer,Double>>> allPossibleRotations = getAllCombinations(rotations);

            List<ValuePair<Integer,Double>> returnPair;
            float highestCorrelation = 0.0f;
            int highestCorrelatedPair = -1;

            //Brute force search each option, first all positive inclinations, then all negative

            for (int i = 0; i < allPossibleRotations.size(); i++) {
                float correlation = getDirectCorrelation(reference, getDoubleRotatedImage(moving, reference, allPossibleRotations.get(i)));
                if(correlation > highestCorrelation){
                    highestCorrelation = correlation;
                    highestCorrelatedPair = i;
                }
            }

            return allPossibleRotations.get(highestCorrelatedPair);
        }

        private List<List<ValuePair<Integer,Double>>> getAllCombinations(MutableTriple<Integer, Double, Double> rotations){
            List<List<ValuePair<Integer,Double>>> fullList = new ArrayList<>();

            //First Double in MutableTriple is inclination degree
            //remaining contains inclination axis
            List<Integer> remaining = IntStream.rangeClosed(0,2).boxed().collect(Collectors.toList());
            remaining.remove(rotations.getLeft());

            //Left and Right ALWAYS get paired.
            fullList.add(getOrderedRotationList(rotations.getLeft(), rotations.getRight(), remaining.get(0), rotations.getMiddle()));
            fullList.add(getOrderedRotationList(rotations.getLeft(), rotations.getRight(), remaining.get(1), rotations.getMiddle()));

            fullList.add(getOrderedRotationList(remaining.get(0), rotations.getMiddle(), rotations.getLeft(), rotations.getRight()));
            fullList.add(getOrderedRotationList(remaining.get(1), rotations.getMiddle(), rotations.getLeft(), rotations.getRight()));

            fullList.add(getOrderedRotationList(rotations.getLeft(), rotations.getRight(), remaining.get(0), -rotations.getMiddle()));
            fullList.add(getOrderedRotationList(rotations.getLeft(), rotations.getRight(), remaining.get(1), -rotations.getMiddle()));

            fullList.add(getOrderedRotationList(remaining.get(0), -rotations.getMiddle(), rotations.getLeft(), rotations.getRight()));
            fullList.add(getOrderedRotationList(remaining.get(1), -rotations.getMiddle(), rotations.getLeft(), rotations.getRight()));

            return fullList;
        }

        private List<ValuePair<Integer,Double>> getOrderedRotationList(Integer axis1, double rotation1, Integer axis2, double rotation2){
            List<ValuePair<Integer,Double>> rotationList = new ArrayList<ValuePair<Integer,Double>>(2);
            rotationList.add(new ValuePair<>(axis1, rotation1));
            rotationList.add(new ValuePair<>(axis2, rotation2));

            return rotationList;
        }

        private RandomAccessibleInterval<T> getDoubleRotatedImage(RealTransformRealRandomAccessible<T, AffineTransform3D> image, Interval interval, List<ValuePair<Integer,Double>> orderedRotations){
            return Views.interval(
                Views.raster(
                        RealViews.transform(image, getDoubleRotatedTransform(orderedRotations))
                ), interval);
        }

        private AffineTransform3D getDoubleRotatedTransform(List<ValuePair<Integer,Double>> orderedRotations){
            AffineTransform3D returnTransform = new AffineTransform3D();
            returnTransform.identity();

            returnTransform.rotate(orderedRotations.get(0).getA(), orderedRotations.get(0).getB());
            returnTransform.rotate(orderedRotations.get(1).getA(), orderedRotations.get(1).getB());

            return returnTransform;
        }

        private float getDirectCorrelation(RandomAccessibleInterval<T> reference, RandomAccessibleInterval<T> moving){
            //Img<FloatType> multiplied = (Img<FloatType>)ops.op("create.img").input(reference, new FloatType()).apply();
            //SciJava ops "math.mul" doesn't work here.
            //ops.op("math.mul").input(reference, Views.interval(Views.extendZero(moving), reference)).output(multiplied).compute();

            IterableInterval multiplied = ijOps.math().multiply(reference,Views.interval(Views.extendZero(moving), reference));

            DoubleType result = (DoubleType) ops.op("stats.sum").input(multiplied).apply();

            return result.getRealFloat();
        }

        private RandomAccessibleInterval rotateImage(RandomAccessibleInterval input, AffineTransform3D transform) {
            return getCurrentWorkingIntervalView(
                    RealViews.transform(extendZeroAndInterpolate(input), transform),
                    input
            );
        }

        private void setSingleAxisCorrelationImage(Img<FloatType> correlationImage, RandomAccessibleInterval<? extends RealType> reference, RandomAccessibleInterval<? extends RealType> moving) {
            FFTConvolution fftConvolution;
            ExecutorService service = Executors.newCachedThreadPool();

//            uiService.show("Reference spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(reference)), 0, 2));
//            uiService.show("Moving spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(moving)), 0, 2));

            fftConvolution = new FFTConvolution(Views.extendPeriodic(convertToSphericalCoordinates(reference)), correlationImage, Views.extendPeriodic(convertToSphericalCoordinates(moving)), correlationImage, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(correlationImage, new ComplexFloatType()).apply(), service);
            fftConvolution.setComputeComplexConjugate(true);
            fftConvolution.setOutput(correlationImage);
            fftConvolution.convolve();
        }


        private MutableTriple<Integer, Double, Double> findHighestRotationalPointAmong(HashMap<Integer, Img<FloatType>> toSearch) {
            //returns dimension and azimuthal angle for rotation
            boolean repeatSearch = true;
            float max;
            MutableTriple<Integer, Long, Long> maxPoint = new MutableTriple<>(-1, -1L, -1L);
            double fractionalIncline = 0.0;
            double fractionalAzimuth = 0.0;

            while(repeatSearch && !toSearch.isEmpty()) {
                max = 0.0F;
                maxPoint = new MutableTriple<>(-1, -1L, -1L);

                for (Integer i : toSearch.keySet()) {

                    RandomAccessibleInterval<FloatType> rotationalCorrelation =
                            Views.hyperSlice(toSearch.get(i), 0, (toSearch.get(i).dimension(0) - 1) / 2);
                    Cursor<FloatType> cursor = rotationalCorrelation.localizingCursor();

                    while (cursor.hasNext()) {
                        if (cursor.next().get() > max) {
                            max = cursor.get().get();
                            maxPoint = new MutableTriple<>(i, cursor.getLongPosition(0), cursor.getLongPosition(1));
                        }
                    }
                }

                fractionalIncline = (maxPoint.getMiddle() - ((toSearch.get(maxPoint.getLeft()).dimension(1) - 1) / 2.0)) / (toSearch.get(maxPoint.getLeft()).dimension(1) - 1);
                fractionalAzimuth = (maxPoint.getRight() - ((toSearch.get(maxPoint.getLeft()).dimension(2) - 1) / 2.0)) / (toSearch.get(maxPoint.getLeft()).dimension(2) - 1);
                if (fractionalIncline < Double.MIN_VALUE && fractionalAzimuth < Double.MIN_VALUE){
                    //Check if max correlation is at zero-rotation, and repeat search if so.
                    toSearch.remove(maxPoint.getLeft());
                    repeatSearch = true;
                }
                else{
                    repeatSearch = false;
                }
            }
            if (maxPoint.getLeft() == -1) {
                return new MutableTriple<>(0, 0.0, 0.0);
            }

            logService.info("Brightest point in Rotational Correlation images, from perspective of axis " + maxPoint.getLeft() +
                    ":\n\tInclination: " + Math.toDegrees(Math.PI * fractionalIncline) + "\n\tAzimuth: " + Math.toDegrees(2 * Math.PI * fractionalAzimuth));

            return new MutableTriple<>(maxPoint.getLeft(), Math.PI * fractionalIncline, 2 * Math.PI * fractionalAzimuth);
        }



        private RandomAccessibleInterval convertToSphericalCoordinates(RandomAccessibleInterval input) {

            RealTransformRealRandomAccessible sphericalInput = RealViews.transform(
                    RealViews.transformReal(extendZeroAndInterpolate(input), SphericalToCartesianTransform3D.getInstance().inverse()),
                    new Scale3D(1, rotationalScaleFactor, rotationalScaleFactor)
            );

            return Views.interval(
                    Views.raster(
                            sphericalInput
                    ),
                    sphericalInterval
            );
        }
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
            logService.info("Translating scaled moving image by: " + translation[0] + "," +translation[1] + "," +translation[2]);
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

//            uiService.show("Crosscorrelated image", Views.hyperSlice(crossCorrelation, 2, (crossCorrelation.dimension(2)-1)/2));

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
