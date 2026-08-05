import io.scif.config.SCIFIOConfig;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imglib2.*;
import net.imglib2.algorithm.fft2.FFTConvolution;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.*;
import net.imglib2.realtransform.interval.IntervalSamplingMethod;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.operators.SetZero;
import net.imglib2.util.Intervals;
import net.imglib2.util.ValuePair;
import net.imglib2.view.ExtendedRandomAccessibleInterval;
import net.imglib2.view.Views;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.DynamicCommand;
import org.scijava.log.LogService;
import org.scijava.ops.api.OpEnvironment;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.scijava.ItemVisibility.MESSAGE;

@Plugin(type = Command.class, headless = true, menuPath = "Process>Registration>Batch 3D Registration")
public class Batch_3D_Registration <T extends RealType<?> & SetZero> extends DynamicCommand {

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

    @Parameter (visibility = MESSAGE, required=false)
    protected String msg = "Set a reference image that moving images will align to:";

    @Parameter(label = "Reference image: ")
    protected File referenceFile;

    @Parameter(label = "Moving images: ")
    protected File[] movingFiles;

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



        ExtendedRandomAccessibleInterval extendedReferenceImage = (ExtendedRandomAccessibleInterval) ops.op("transform.extendZeroView").input(referenceImage).apply();
        RealRandomAccessible interpolatedReference = Views.interpolate(extendedReferenceImage, new NLinearInterpolatorFactory());

        offsetReferenceView = RealViews.transform(interpolatedReference, originOffset);

        rotator = new Rotation(getCurrentWorkingIntervalView(offsetReferenceView, originOffset, referenceImage));

    }

    @Override
    public void run(){
        Dataset referenceImage;

        if(!datasetIOService.canOpen(referenceFile.getPath())){
            throw new RuntimeException(new IOException("Cannot open reference file as image."));
        }
        try {
            referenceImage = datasetIOService.open(referenceFile.getPath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        errorChecking(referenceImage);

        initializePlugin(referenceImage);



        //Can process these here for all resolutions of reference image, to speed up processing:
        //spherical projection, and 3 separate axis oriented area-matched azimuthal projections, all in Real space (need to figure out appropriate data structure)
        //log-polar transform of 3 azimuthal projections,



        for (File movingFile:movingFiles){

            //Set up the running combined transform
            AffineTransform3D currentTransform = new AffineTransform3D();
            currentTransform.set(originOffset);

            if(!datasetIOService.canOpen(movingFile.getPath())){
                logService.warn("Skipping non-openable moving file: " + movingFile.getPath());
                continue;
            }
            Dataset movingImage;
            try {
                movingImage = datasetIOService.open(movingFile.getPath());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            ExtendedRandomAccessibleInterval extendedMovingImage = (ExtendedRandomAccessibleInterval) ops.op("transform.extendZeroView").input(movingImage).apply();
            RealRandomAccessible interpolatedMoving = Views.interpolate(extendedMovingImage, new NLinearInterpolatorFactory());

            //for loop for sampling iterations starts here; do a test run with up to 5 loops at FULL sample rate, and save images at each iteration

            for (int i = 0; i < 5; i++) {

                RealTransformRandomAccessible currentView = RealViews.transform(interpolatedMoving, currentTransform);

                //calculate  and appplyTranslation
                currentTransform.translate(
                        translationViaCenterOfGravity(
                                getCurrentWorkingIntervalView(offsetReferenceView, originOffset, referenceImage),
                                getCurrentWorkingIntervalView(currentView, currentTransform, movingImage)
                        )
                );
                //currentTransform.translate(translator.findMovingImageTranslation(referenceImage, getCurrentZeroMinWorkingIntervalView(currentView, movingImage)));
                currentView = RealViews.transform(interpolatedMoving, currentTransform);

                uiService.show(movingImage.getName(), getCurrentZeroMinWorkingIntervalView(currentView, referenceImage));

                uiService.show(referenceImage.getName(), getCurrentZeroMinWorkingIntervalView(offsetReferenceView, referenceImage));

                currentTransform.concatenate(rotator.findFullImageRotation(offsetReferenceView, currentView));

                uiService.show(movingImage.getName(), getCurrentZeroMinWorkingIntervalView(currentView, referenceImage));

                uiService.show(referenceImage.getName(), getCurrentZeroMinWorkingIntervalView(offsetReferenceView, referenceImage));

                Dataset output = datasetService.create(
                        getCurrentZeroMinWorkingIntervalView(RealViews.transform(interpolatedMoving, currentTransform), referenceImage)
                );

                try {
                    datasetIOService.save(output, saveFolder.getPath() + File.separator + "Registered-" + i + "-" + movingImage.getName() + ".tif", config);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }


            }

            Dataset output = datasetService.create(
                getCurrentZeroMinWorkingIntervalView(RealViews.transform(interpolatedMoving, currentTransform), referenceImage)
            );

            try {
                datasetIOService.save(output, saveFolder.getPath() + File.separator + "Registered-" + movingImage.getName() + ".tif", config);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }
    }

    private RandomAccessibleInterval getCurrentWorkingIntervalView(RealTransformRealRandomAccessible view, AffineTransform3D transformation, Interval originalInterval){
        return Views.interval(
                Views.raster(
                        view
                ),
                Intervals.smallestContainingInterval(
                        transformation.boundingInterval(originalInterval, IntervalSamplingMethod.CORNERS)
                )
        );
    }

    private RandomAccessibleInterval getCurrentZeroMinWorkingIntervalView(RealTransformRealRandomAccessible view, Interval interval){
        return Views.interval(
                    Views.raster(
                        RealViews.transform(view, originOffset.inverse())
                    ),
                interval
        );
    }


    public double[] translationViaCenterOfGravity(RandomAccessibleInterval reference, RandomAccessibleInterval moving){
        double[] referenceCenter = ((RealPoint) ops.op("geom.centerOfGravity").input(reference).apply()).positionAsDoubleArray();
        double[] movingCenter = ((RealPoint) ops.op("geom.centerOfGravity").input(moving).apply()).positionAsDoubleArray();


        double[] translation = new double[referenceCenter.length];

        for (int i = 0; i < referenceCenter.length; i++) {
            translation[i] = referenceCenter[i]-movingCenter[i];
        }
        System.out.println(translation[0] + "," + translation[1] + "," +translation[2]);
        return translation;
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

//    public class Rotation{
//        Interval interval;
//
//        public Rotation(Interval inputInterval) {
//            //todo: move this
//
//            double[] max = inputInterval.maxAsDoubleArray();
//
//            double intervalDistance = Math.sqrt(
//                    Math.pow(max[0], 2) + Math.pow(max[1], 2) + Math.pow(max[2], 2)
//            );
//
//            interval = new FinalInterval(new long[]{0, -314, -314},
//                    new long[]{(long)Math.floor(intervalDistance), 314, 314});
////
//        }
//
//        public AffineTransform3D findFullImageRotation(RealTransformRandomAccessible reference, RealTransformRandomAccessible moving){
//            RealTransformRealRandomAccessible sphericalReference = RealViews.transformReal(reference, SphericalToCartesianTransform3D.getInstance().inverse());
//            RealTransformRealRandomAccessible sphericalMoving = RealViews.transformReal(moving, SphericalToCartesianTransform3D.getInstance().inverse());
//
////            uiService.show("Spherical Ref", ImgView.wrap(Views.zeroMin(Views.interval(getSphericalRA(sphericalReference), interval))));
////            uiService.show("Moving Ref", ImgView.wrap(Views.zeroMin(Views.interval(getSphericalRA(sphericalMoving), interval))));
//
//            Dimensions dims = new FinalDimensions(interval.dimensionsAsLongArray());
//            Img<FloatType> rotationalCC = (Img<FloatType>) ops.op("create.img").input(dims, new FloatType()).apply();
//
//            ExecutorService service = Executors.newCachedThreadPool();
//
//            FFTConvolution fftConvolution = new FFTConvolution<>(getSphericalRA(sphericalReference), interval, getSphericalRA(sphericalMoving), interval, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(interval, new ComplexFloatType()).apply(), service );
//
//            fftConvolution.setComputeComplexConjugate(true);
//            fftConvolution.setOutput(rotationalCC);
//            fftConvolution.convolve();
//
//            uiService.show(Views.hyperSlice(rotationalCC, 0, Math.round(interval.max(0)/2.0) ));
//
//            findMaxPoint(Views.hyperSlice(rotationalCC, 0, Math.round(interval.max(0)/2.0) ));
//
////            uiService.show("CC", rotationalCC);
//            return new
//
//        }
//
//        private double[] findMaxPoint(IterableInterval<FloatType> image) {
//            float max = 0;
//            double[] maxPoint = new double[image.numDimensions()];
//            Cursor<FloatType> cursor = image.localizingCursor();
//
//            while (cursor.hasNext()){
//                if(cursor.next().get() > max){
//                    max = cursor.get().get();
//                    maxPoint = cursor.positionAsDoubleArray();
//                }
//            }
//            return maxPoint;
//        }
//
//        private RandomAccessible getSphericalRA(RealTransformRealRandomAccessible image){
//            AffineTransform3D scale100 = new AffineTransform3D();
//            scale100.identity();
//            scale100.scale(100);
//            return Views.raster(
//                            RealViews.transformReal(image, scale100)
//                    );
//
//        }
//    }

    public class Rotation{
        //class for Rotation via flat Sum intensity projection

        Interval interval;
        Double searchDistance;
        Long sampleCount;

        double sampleAngle;
        AffineTransform3D rotate0to2;
        AffineTransform3D rotate1to2;
        Dimensions targetDims;


        public Rotation(Interval inputInterval){
            rotate0to2.set( 0,0,1,0,
                    0,1,0,0,
                    -1,0,0,0);

            rotate1to2.set( 1,0,0,0,
                    0,0,-1,0,
                    0,1,0,0);
        }

    public AffineTransform3D findFullImageRotation(RealTransformRandomAccessible reference, RealTransformRandomAccessible moving){
        AffineTransform3D rotation = new AffineTransform3D();
        rotation.identity();
        HashMap<Integer,Img<FloatType>> axesToSearch = new HashMap<>(reference.numDimensions());
        for (int i = 0; i < reference.numDimensions(); i++) {
            axesToSearch.put(i, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(targetDims), new FloatType()).apply());
        }

        while (!axesToSearch.isEmpty()){
            RealTransformRealRandomAccessible currentView = RealViews.transformReal(moving, rotation);

            ValuePair<Integer,Double> toRotate = findHighestRotationCorrelation(reference,currentView, axesToSearch);

            System.out.println("Rotating " + toRotate.getA() + " by " + toRotate.getB());
            rotation.rotate(toRotate.getA(), toRotate.getB());
            axesToSearch.remove(toRotate.getA());
        }
        return rotation;
    }

    public ValuePair<Integer, Double> findHighestRotationCorrelation(RealTransformRealRandomAccessible reference, RealTransformRealRandomAccessible moving, HashMap<Integer,Img<FloatType>> toSearch){
        for (Integer axis: toSearch.keySet()){
            switch (axis){
                case 0:
                    setSingleAxisCorrelationImage(toSearch.get(0), RealViews.transformReal(reference, rotate0to2), RealViews.transformReal(moving, rotate0to2));
//                        uiService.show("X-axis corr", toSearch.get(0));
                    break;
                case 1:
                    setSingleAxisCorrelationImage(toSearch.get(1), RealViews.transformReal(reference, rotate1to2), RealViews.transformReal(moving, rotate1to2));
//                        uiService.show("Y-axis corr", toSearch.get(1));
                    break;
                case 2:
                    setSingleAxisCorrelationImage(toSearch.get(2), reference, moving);
//                        uiService.show("Z-axis corr", toSearch.get(2));
                    break;

            }
        }

        return findHighestRotationalPointAmong(toSearch);
    }

    private void setSingleAxisCorrelationImage(Img<FloatType> correlationImage, RealTransformRealRandomAccessible reference, RealTransformRealRandomAccessible moving){
        //create sum intensity projection for both images
        Img<FloatType> referencePolarAzimuth = (Img<FloatType>) ops.op("transform.project").input(reference).apply();
        Img<FloatType> movingPolarAzimuth = (Img<FloatType>) ops.op("transform.project").input(moving).apply();

        polarAzimuthCompute(referencePolarAzimuth, reference);
        polarAzimuthCompute(movingPolarAzimuth, moving);

//            uiService.show("Ref polar",referencePolarAzimuth);
//            uiService.show("Moving polar",movingPolarAzimuth);

        FFTConvolution fftConvolution;
        ExecutorService service = Executors.newCachedThreadPool();


        fftConvolution = new FFTConvolution(Views.extendPeriodic(referencePolarAzimuth), correlationImage,Views.extendPeriodic(movingPolarAzimuth), correlationImage, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(correlationImage, new ComplexFloatType()).apply(), service);
        fftConvolution.setComputeComplexConjugate(true);
        fftConvolution.setOutput(correlationImage);
        fftConvolution.convolve();
    }





    private ValuePair<Integer, Double> findHighestRotationalPointAmong(HashMap<Integer,Img<FloatType>> toSearch){
        float max = 0.0F;
        ValuePair<Integer,Point> maxPoint;
        maxPoint = new ValuePair<>(0, new Point(new long[]{0,0,0}));

        for (Integer i: toSearch.keySet()) {
            Cursor<FloatType> cursor = toSearch.get(i).localizingCursor();

            while (cursor.hasNext()){
                if(cursor.next().get() > max){
                    max = cursor.get().get();
                    maxPoint = new ValuePair<>(i,cursor.positionAsPoint());
                }
            }
        }
        double fractionalAngle = (maxPoint.getB().getDoublePosition(1) - ((toSearch.get(maxPoint.getA()).dimension(1)-1)/2.0))/toSearch.get(maxPoint.getA()).dimension(1);

        return new ValuePair<>(maxPoint.getA(), 2*Math.PI*fractionalAngle);
    }

    }

//    public class Rotation{
//        //class for Rotation via azimuthal projection, first attempt, failed
//
//        Interval interval;
//        Double searchDistance;
//        Long sampleCount;
//
//        double sampleAngle;
//        AffineTransform3D rotate0to2;
//        AffineTransform3D rotate1to2;
//        Dimensions targetDims;
//
//
//        public Rotation(Interval inputInterval){
//            //todo: move this
//            targetDims = new FinalDimensions(new long[]{361, 721});
//
//            interval = new FinalInterval(inputInterval);
//            double[] max = interval.maxAsDoubleArray();
//
//            searchDistance = Math.sqrt(
//                    Math.pow(max[0], 2) + Math.pow(max[1], 2) + Math.pow(max[2], 2)
//            );
//
//            sampleCount = 50L;
//
//            rotate0to2 = new AffineTransform3D();
//            rotate1to2 = new AffineTransform3D();
//
//            //todo: these are based off of Cartesian coordinates, need them to be based off spherical coordinates?
//            //x-axis to z-axis
//
//            //Below are Cartesian coordinate rotation matricies
//            rotate0to2.set( 0,0,1,0,
//                    0,1,0,0,
//                    -1,0,0,0);
//
//            rotate1to2.set( 1,0,0,0,
//                    0,0,-1,0,
//                    0,1,0,0);
//        }
//
//        public void setRotationParameters(double sampleAngle){
//
//
//
//        }
//
//
//
//        public AffineTransform3D findFullImageRotation(RealTransformRandomAccessible reference, RealTransformRandomAccessible moving){
//            AffineTransform3D rotation = new AffineTransform3D();
//            rotation.identity();
//            HashMap<Integer,Img<FloatType>> axesToSearch = new HashMap<>(reference.numDimensions());
//            for (int i = 0; i < reference.numDimensions(); i++) {
//                axesToSearch.put(i, (Img<FloatType>) ops.op("create.img").input(new FinalDimensions(targetDims), new FloatType()).apply());
//            }
//
//            while (!axesToSearch.isEmpty()){
//                RealTransformRealRandomAccessible currentView = RealViews.transformReal(moving, rotation);
//
//                ValuePair<Integer,Double> toRotate = findHighestRotationCorrelation(reference,currentView, axesToSearch);
//
//                System.out.println("Rotating " + toRotate.getA() + " by " + toRotate.getB());
//                rotation.rotate(toRotate.getA(), toRotate.getB());
//                axesToSearch.remove(toRotate.getA());
//            }
//            return rotation;
//        }
//
//        public ValuePair<Integer, Double> findHighestRotationCorrelation(RealTransformRealRandomAccessible reference, RealTransformRealRandomAccessible moving, HashMap<Integer,Img<FloatType>> toSearch){
//            for (Integer axis: toSearch.keySet()){
//                switch (axis){
//                    case 0:
//                        setSingleAxisCorrelationImage(toSearch.get(0), RealViews.transformReal(reference, rotate0to2), RealViews.transformReal(moving, rotate0to2));
////                        uiService.show("X-axis corr", toSearch.get(0));
//                        break;
//                    case 1:
//                        setSingleAxisCorrelationImage(toSearch.get(1), RealViews.transformReal(reference, rotate1to2), RealViews.transformReal(moving, rotate1to2));
////                        uiService.show("Y-axis corr", toSearch.get(1));
//                        break;
//                    case 2:
//                        setSingleAxisCorrelationImage(toSearch.get(2), reference, moving);
////                        uiService.show("Z-axis corr", toSearch.get(2));
//                        break;
//
//                }
//            }
//
//            return findHighestRotationalPointAmong(toSearch);
//        }
//
//        private void setSingleAxisCorrelationImage(Img<FloatType> correlationImage, RealTransformRealRandomAccessible reference, RealTransformRealRandomAccessible moving){
//            //create Polar, or log-polar, img, for both
//            Img<FloatType> referencePolarAzimuth = (Img<FloatType>) ops.op("create.img").input(correlationImage).apply();
//            Img<FloatType> movingPolarAzimuth = (Img<FloatType>) ops.op("create.img").input(correlationImage).apply();
//
//            polarAzimuthCompute(referencePolarAzimuth, reference);
//            polarAzimuthCompute(movingPolarAzimuth, moving);
//
////            uiService.show("Ref polar",referencePolarAzimuth);
////            uiService.show("Moving polar",movingPolarAzimuth);
//
//            FFTConvolution fftConvolution;
//            ExecutorService service = Executors.newCachedThreadPool();
//
//
//            fftConvolution = new FFTConvolution(Views.extendPeriodic(referencePolarAzimuth), correlationImage,Views.extendPeriodic(movingPolarAzimuth), correlationImage, (ImgFactory<ComplexFloatType>) ops.op("create.imgFactory").input(correlationImage, new ComplexFloatType()).apply(), service);
//            fftConvolution.setComputeComplexConjugate(true);
//            fftConvolution.setOutput(correlationImage);
//            fftConvolution.convolve();
//        }
//
//        private void polarAzimuthCompute(Img<FloatType> output,RealTransformRealRandomAccessible input){
//            //Output image origin: center at 0 degrees polar coordinates.
//            RealTransformRealRandomAccessible sphericalInput = RealViews.transformReal(input, SphericalToCartesianTransform3D.getInstance().inverse());
//            //iterate through each pixel, at each, find corresponding azimuthal and polar angles, sum project that
//            Cursor<FloatType> cursor = output.localizingCursor();
//            while(cursor.hasNext()){
//                cursor.fwd();
//                ValuePair<Double, Double> angles = getSpericalAngles((double)cursor.getLongPosition(0)/ (double)output.dimension(0), ((double)cursor.getLongPosition(1)/ (double)output.dimension(1)));
//                cursor.get().set(sumSphericalProjection(sphericalInput, Math.log(angles.getA()) , angles.getB()));
//            }
//        }
//
//        //needs to take in x<distance from center in azimuthal image>, fractional rotation of azimuthal angle (y), and return polar, azimuth
//        private ValuePair<Double,Double> getSpericalAngles(double fractionalR, double fractionalAzimuth){
//            Double azimuth = 2*Math.PI*fractionalAzimuth-Math.PI;
//
//            //definitely not sure if this is correct, relies on wikipedia page.
//
//            //Source of GOOD equations: Map projections - A working manual; Page 198 of https://pubs.usgs.gov/pp/1395/report.pdf
//            Double inclination = 2*Math.asin(fractionalR);
//
//            return new ValuePair<>(inclination,azimuth);
//        }
//
//        private Float sumSphericalProjection(RealTransformRealRandomAccessible<? extends RealType, ? extends RealTransform> input, double inclination, double azimuth){
//            float sum = 0.0F;
//            for (int i = 0; i < sampleCount; i++) {
//                sum += input.getAt(i*(sampleCount/searchDistance), inclination, azimuth).getRealFloat();
//            }
//            return sum;
//        }
//
//        private ValuePair<Integer, Double> findHighestRotationalPointAmong(HashMap<Integer,Img<FloatType>> toSearch){
//            float max = 0.0F;
//            ValuePair<Integer,Point> maxPoint;
//            maxPoint = new ValuePair<>(0, new Point(new long[]{0,0,0}));
//
//            for (Integer i: toSearch.keySet()) {
//                Cursor<FloatType> cursor = toSearch.get(i).localizingCursor();
//
//                while (cursor.hasNext()){
//                    if(cursor.next().get() > max){
//                        max = cursor.get().get();
//                        maxPoint = new ValuePair<>(i,cursor.positionAsPoint());
//                    }
//                }
//            }
//            double fractionalAngle = (maxPoint.getB().getDoublePosition(1) - ((toSearch.get(maxPoint.getA()).dimension(1)-1)/2.0))/toSearch.get(maxPoint.getA()).dimension(1);
//
//            return new ValuePair<>(maxPoint.getA(), 2*Math.PI*fractionalAngle);
//        }
//    }


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

        private Dimensions greatestOddDimensions(Interval input1, Interval input2){
            long[] dim1 = input1.dimensionsAsLongArray();
            long[] dim2 = input2.dimensionsAsLongArray();

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
