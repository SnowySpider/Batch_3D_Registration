/*-
 * #%L
 * Scijava plugin for automatic 3D registration
 * %%
 * Copyright (C) 2019 - 2026 Andrew McCall, University at Buffalo
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

package batch3Dregistration;

import com.google.common.collect.EvictingQueue;
import io.scif.config.SCIFIOConfig;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImgPlus;
import net.imagej.axis.Axes;
import net.imagej.ops.OpService;
import net.imglib2.*;
import net.imglib2.RandomAccess;
import net.imglib2.algorithm.fft2.FFTConvolution;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.ImgView;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.realtransform.*;
import net.imglib2.realtransform.interval.IntervalSamplingMethod;
import net.imglib2.roi.*;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegion;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.type.logic.BitType;
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
import org.scijava.command.DynamicCommand;
import org.scijava.legacy.service.OpEnvironmentService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import outofboundsspherical.OutOfBoundsSpherical3DFactory;

import static org.scijava.ItemVisibility.MESSAGE;

@Plugin(type = DynamicCommand.class, headless = true, menuPath = "Plugins>Registration>Batch 3D Registration")
public class Batch_3D_Registration <T extends RealType<T>, F extends FloatType, R extends InvertibleRealTransform> implements Command {

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
    protected OpEnvironmentService ops;

    @Parameter
    protected OpService ijOps;

    @Parameter (visibility = MESSAGE, required=false)
    protected String msg = "Set a reference image that all moving images will align to:";

    @Parameter(label = "Reference image: ")
    protected File referenceFile;

    @Parameter(label = "Channel for correlation (ignore for single channel images): ")
    protected Integer correlationChannel;

    @Parameter(label = "Reference mask (optional): ", required = false)
    protected File maskFile;

    @Parameter(label = "Registration accuracy: ")
    protected B3dParameters.Resolutions resolutionChoice = B3dParameters.Resolutions.Normal;

    @Parameter(label = "Moving images: ")
    protected File[] movingFiles;

    @Parameter(label = "Threshold (Raw pixel value): ", required = false)
    protected Double thresholdValue;

    @Parameter(label = "Registered images output directory: ", style = "directory")
    protected File saveFolder;

    protected SCIFIOConfig config;

    protected Dataset origReferenceImage;
    protected RandomAccessibleInterval<FloatType> referenceImage;
    protected Converter floatMinusMean;
    protected Img<BitType> referenceMask;
    protected boolean maskApplied;
    protected CorrelationTranslation translator;
    protected Rotation2D rotator;
    protected AffineTransform3D originOffset;
    protected RealTransformRandomAccessible offsetReferenceView;
    protected LinkedList<B3dParameters> resolutionList;
    protected float mean;

    protected void errorChecking(Dataset testImage){
        if(testImage.dimensionIndex(Axes.TIME) > 1){
            throw new RuntimeException(new IOException("This plugin does not currently support time-lapse images."));
        }
    }

    protected void initializePlugin() throws IOException {
        if(!datasetIOService.canOpen(referenceFile.getPath())){
            throw new RuntimeException(new IOException("Cannot open reference file as image."));
        }
        logService.info("Opening reference image file: " + referenceFile.getPath());

        try {
            origReferenceImage = datasetIOService.open(referenceFile.getPath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        errorChecking(origReferenceImage);

        if(maskFile == null || maskFile.getPath().isEmpty()){
            referenceMask = null;
            maskApplied = true;
        }
        else{
            try{
                {
                    referenceMask = (Img) ops.env().op("convert.bit").input(
                            datasetIOService.open(maskFile.getPath())
                    ).apply();
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            if(referenceMask.numDimensions() != 3){
                throw new RuntimeException(new IOException("Reference mask must have exactly 3 dimensions."));
            }
            if (origReferenceImage.size() != referenceMask.size()){
                throw new RuntimeException(new IOException("Reference mask dimensions do not match reference image."));
            }
            maskApplied = false;
        }

        correlationChannel -= 1;

        if(saveFolder != null) saveFolder.mkdirs();

        config = new SCIFIOConfig();
        config.writerSetFailIfOverwriting(false);

        translator = new CorrelationTranslation();

        RandomAccessibleInterval<? extends RealType> tempRef = getWorkingRAI(origReferenceImage);
        mean = ijOps.stats().mean(tempRef).getRealFloat();
        floatMinusMean = (Converter<RealType, FloatType>) (input, output) -> output.setReal(input.getRealFloat() - mean);
        referenceImage = Converters.convert(tempRef, floatMinusMean, new FloatType());

        logService.info("Moving reference image center to origin");
        originOffset = centerOfGravityToOrigin(tempRef);

        resolutionList = new LinkedList<>();

        switch (resolutionChoice){
            case Native:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Native));
                break;
            case VeryHigh:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.VeryHigh));
                break;
            case High:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.High));
                break;
            case Normal:
            default:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Normal));
                break;
            case Moderate:
                resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Moderate));
            case Low:
            case VeryLow:
        }
        if(resolutionChoice != B3dParameters.Resolutions.VeryLow)
            resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.Low));
        resolutionList.addFirst(new B3dParameters(referenceImage, B3dParameters.Resolutions.VeryLow));
        resolutionList.addFirst(new B3dParameters(referenceImage, 75000, 15.0, false));
        resolutionList.addFirst(new B3dParameters(referenceImage, 50000, 30.0, false));
    }

    protected RandomAccessibleInterval<? extends RealType> getWorkingRAI(Dataset input){
        RandomAccessibleInterval<? extends RealType> returnRAI = input;

        applyThreshold(returnRAI,thresholdValue);
        int chIndex = input.dimensionIndex(Axes.CHANNEL);

        if(input.dimension(chIndex) > 1){
            returnRAI = Views.dropSingletonDimensions(
                (RandomAccessibleInterval)ops.env().op("transform.hyperSliceView")
                .input(returnRAI, chIndex, correlationChannel).apply()
            );
        }

        if(maskFile != null && !maskFile.getPath().isEmpty()){
            maskApplied = false;
        }

        return returnRAI;
    }

    @Override
    public void run(){
        logService.info("Starting Batch 3D Registration plugin");

        logService.info("Initializing plugin");
        try {
            initializePlugin();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

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
            errorChecking(originalMovingImage);

            RandomAccessibleInterval<FloatType> currentReference = referenceImage;

            offsetReferenceView = RealViews.transform(extendConstAndInterpolate(currentReference, mean), originOffset);

            //Need to create a copy, so we can save the original image.
            logService.info("Applying threshold to moving image");

            RandomAccessibleInterval tempMoving = getWorkingRAI(originalMovingImage);
            //Set up the running combined transform
            AffineTransform3D currentTransform = centerOfGravityToOrigin(tempMoving);
            RandomAccessibleInterval movingImage = Converters.convert(tempMoving, floatMinusMean, new FloatType());
            RealRandomAccessible extendedInterpolatedMoving = extendConstAndInterpolate(movingImage, mean);

            for(B3dParameters resolution:resolutionList) {

                logService.info("--Starting alignment at " + resolution.getSubsampleScale()[0] + " scaling factor, and "
                + (180/(resolution.getRotationalScaleFactor()*Math.PI)) + " degrees rotational resolution.");

                RealTransformRandomAccessible currentMovingView = RealViews.transform(extendedInterpolatedMoving, currentTransform);

                if(!maskApplied && resolution.getTranslateByCorrelate()) {
                    logService.info("Applying mask to reference image");
                    ImgLabeling labeledImg = ImgLabeling.fromImageAndLabels(referenceMask, new ArrayList(List.of("target")));
                    LabelRegion region = new LabelRegions(labeledImg).getLabelRegion("target");
                    currentReference = makeRai(region, referenceImage);

                    maskApplied = true;
                }

                RandomAccessibleInterval subSampledReference = getSubsampledWorkingIntervalView(
                        offsetReferenceView,
                        currentReference,
                        resolution.getSubsampleScale()
                );


                RandomAccessibleInterval subSampledMoving = getSubsampledWorkingIntervalView(
                        currentMovingView,
                        movingImage,
                        resolution.getSubsampleScale()
                );

//                showStackedImages("subSampled Centered", getCenteredImage(subSampledReference), getCenteredImage(subSampledMoving));

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


            Dataset output = datasetService.create(
                ImgPlus.wrap(
                    finalMovingTransform(originalMovingImage, currentTransform, referenceImage),
                    originalMovingImage.getImgPlus()
                )
            );

            logService.info("Saving registered image:" + saveFolder.getPath() + File.separator + "Registered-" + originalMovingImage.getName() + ".tif");
            try {
                datasetIOService.save(output, saveFolder.getPath() + File.separator + "Registered-" + originalMovingImage.getName() + ".tif", config);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        logService.info("Batch 3D registration finished.");
    }

    private RandomAccessibleInterval makeRai(LabelRegion region, RandomAccessibleInterval<FloatType> image) {
        Img<FloatType> tinyImg = (Img<FloatType>) ops.env().op("create.img").input(new FinalDimensions(region.dimensionsAsLongArray()), new FloatType()).apply();
        RandomAccessibleInterval<FloatType> output = Views.translate(tinyImg, region.minAsLongArray());

        IterableInterval ii = Regions.sample((IterableRegion<?>) region, image);

        Cursor<FloatType> sampler = ii.localizingCursor();
        RandomAccess<FloatType> targetWriter = output.randomAccess();
        while(sampler.hasNext()){
            sampler.fwd();
            targetWriter.setPositionAndGet(sampler.positionAsLongArray()).set(sampler.get());
        }
        return output;
    }

    private AffineTransform3D scaleTransform(AffineTransform3D input, double[] scale){
        input.set(input.get(0,3)/scale[0],0,3);
        input.set(input.get(1,3)/scale[1],1,3);
        input.set(input.get(2,3)/scale[2],2,3);

        return input;
    }

    private Img finalMovingTransform(Dataset moving, AffineTransform3D transform, Interval interval){
        final int chIndex = moving.dimensionIndex(Axes.CHANNEL);   // e.g. 3 for X,Y,Z,C
        final long nChannels = moving.dimension(chIndex);

        if(chIndex == -1){
            return ImgView.wrap(getCurrentWorkingIntervalView(
                RealViews.transform(
                    RealViews.transform(
                        Views.interpolate(
                                (ExtendedRandomAccessibleInterval)ops.env().op("transform.extendZeroView").input(moving).apply(),new NLinearInterpolatorFactory()
                        ),
                        transform
                    ),
                    originOffset.inverse()
                ),
                interval
            ));
        }

        Img out = moving.getImgPlus().factory().create(interval);

        for (long c = 0; c < nChannels; c++) {
            // 3D view of one channel
            RandomAccessibleInterval src3d = Views.hyperSlice(moving.getImgPlus(), chIndex, c);
            RandomAccessibleInterval<T> dst3d = Views.hyperSlice(out, chIndex, c);

            // Extend (zero outside), interpolate, transform, crop to original bounds
            RealRandomAccessible<T> interp = Views.interpolate(
                    Views.extendZero(src3d), new NLinearInterpolatorFactory());
            RandomAccessibleInterval<T> rotated = Views.interval(
                    RealViews.transform(RealViews.transform(interp, transform),originOffset.inverse()), dst3d);

            // Copy into the output
            LoopBuilder.setImages(rotated, dst3d).multiThreaded().forEachPixel((a, b) -> b.set(a));
        }
        return out;
    }


    private AffineTransform3D alignSubsampledImages(RandomAccessibleInterval<FloatType> referenceSubsampled, RandomAccessibleInterval<FloatType> movingSubsampled, B3dParameters resolutionParams){

//        showStackedImages("Passed in", getCenteredImage(referenceSubsampled), getCenteredImage(movingSubsampled));

        rotator = new Rotation2D(referenceSubsampled, resolutionParams.getRotationalScaleFactor());
        RealRandomAccessible interpolatedReference = extendConstAndInterpolate(referenceSubsampled, mean);

        AffineTransform3D subsampledAlignment = new AffineTransform3D();
        subsampledAlignment.identity();

        RealRandomAccessible<FloatType> interpolatedMoving = extendConstAndInterpolate(movingSubsampled, mean);

        //Need to somehow translate the data, and then make that the center of the interval...
        //for loop for sampling iterations starts here; do a test run with up to 5 loops at FULL sample rate, and save images at each iteration

        Double transformationMagnitude = Double.POSITIVE_INFINITY;
        int i = 0;
        Double transformationMagnitudeTarget = Double.MIN_VALUE;
        EvictingQueue<Double> lastThreeMagnitudes = EvictingQueue.create(4);
        boolean halfRotation = false;

        while(transformationMagnitude > transformationMagnitudeTarget){
            double[] initialTransformation = subsampledAlignment.getRowPackedCopy();

            RealTransformRandomAccessible<FloatType, ? extends InvertibleRealTransform> subsampledMovingView = RealViews.transform(interpolatedMoving, subsampledAlignment);

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

            //currentTransform.translate(translator.findMovingImageTranslation(referenceImage, getCurrentZeroMinWorkingIntervalView(currentMovingView, originalMovingImage)));
            subsampledMovingView = RealViews.transform(interpolatedMoving, subsampledAlignment);
//                uiService.show("Centered reference", Views.zeroMin(Views.interval(offsetReferenceView, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
//                uiService.show("Centered moving",Views.zeroMin(Views.interval(currentMovingView, new FinalInterval(new long[]{-200, -200, -200}, new long[]{200,200,200}))));
//            showStackedImages("Post translation " + i, getCenteredImage(referenceSubsampled), getCenteredImage(getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)));

            AffineTransform3D rotation = rotator.findFullImageRotation(
                    referenceSubsampled,
                    getCurrentWorkingIntervalView(subsampledMovingView, movingSubsampled)
            );

            //todo: report rotation value after halfRotation, not before?
            if(halfRotation){
                AffineTransform3D identity = new AffineTransform3D();
                identity.identity();
                double[] rowPackedRotation = rotation.getRowPackedCopy();
                double[] rowPackedIdentity = identity.getRowPackedCopy();
                for (int j = 0; j < rowPackedRotation.length; j++) {
                    rowPackedRotation[j] = (rowPackedRotation[j] + rowPackedIdentity[j]) /2;
                }
                rotation.set(rowPackedRotation);
                halfRotation = false;
            }

            subsampledAlignment.preConcatenate(rotation);
            subsampledMovingView = RealViews.transform(interpolatedMoving, subsampledAlignment);

            transformationMagnitude = getTransformationMagnitude(initialTransformation,subsampledAlignment.getRowPackedCopy());

            lastThreeMagnitudes.add(transformationMagnitude);

            if(isCircular(lastThreeMagnitudes)){
                halfRotation = true;
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

    //Troubleshooting function to get images cenetered on origin.
    private RandomAccessibleInterval getCenteredImage(RandomAccessibleInterval input){

        return Views.zeroMin(
                Views.interval(
                        input,
                        new FinalInterval(new long[]{-20, -20, -20}, new long[]{20,20,20})
                )
        );
    }

    //Troubleshooting function to show overlay of reference with moving image.
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

    private void applyThreshold(RandomAccessibleInterval<? extends RealType> input, Double thresholdValue){
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

        return Views.interval(
                    Views.raster(
                        RealViews.transform(view, scale3D)
                    ),
                new FinalInterval(min, max)
        );
    }

    private RealRandomAccessible extendConstAndInterpolate(RandomAccessibleInterval<FloatType> input, double value){
        return Views.interpolate(
                Views.extendValue(input, -value),
                new NLinearInterpolatorFactory()
        );
    }



    public AffineTransform3D centerOfGravityToOrigin(RandomAccessibleInterval<? extends RealType> target){

        /* Reference image center of gravity is already at origin, so only need to move the moving image to the origin.
         */
        AffineTransform3D translationAffine = new AffineTransform3D();

        RealPoint center = ((RealPoint) ops.env().op("geom.centerOfGravity").input(
                target
        ).apply());

        translationAffine.set( 1,0,0,-center.getDoublePosition(0),
                0,1,0,-center.getDoublePosition(1),
                0,0,1,-center.getDoublePosition(2));


        logService.info("Translating image by: "
                + (-center.getDoublePosition(0))
                + "," + (-center.getDoublePosition(1))
                + "," + (-center.getDoublePosition(2)));
        return translationAffine;
    }


    /* Class rotates based on correlation of spherical coordinate data. "2D" because it utilizes both the azimuth
     * and the incline dimensions.
     */
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

            long inclinationRange = (long)Math.floor(Math.PI*rotationalScaleFactor);
            inclinationRange += inclinationRange % 2;
            sphericalInterval = new FinalInterval(new long[]{0, 0, -(long)Math.ceil(Math.PI*rotationalScaleFactor)}, new long[]{(long) maxDist, inclinationRange, (long)Math.floor(Math.PI*rotationalScaleFactor)});
        }

        public AffineTransform3D findFullImageRotation(RandomAccessibleInterval<FloatType> reference, RandomAccessibleInterval<FloatType> moving) {
            HashMap<Integer, Img<FloatType>> axesToSearch = new HashMap<>(reference.numDimensions());

//            showStackedImages("Stacked", reference, moving);

            AffineTransform3D currentRotation = new AffineTransform3D();
            currentRotation.identity();
//            uiService.show("Moving in findFullImageRotation", datasetService.create(ImgView.wrap(moving)));
//            showStackedImages("FullImageRotationInput", reference, moving);

//            long[] min = sphericalInterval.minAsLongArray();
//            long[] max = sphericalInterval.maxAsLongArray();
//            min[1] = -max[1];
//            FinalInterval expanded = new FinalInterval(min, max);

            axesToSearch.put(0, (Img<FloatType>) ops.env().op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
            axesToSearch.put(1, (Img<FloatType>) ops.env().op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());
            axesToSearch.put(2, (Img<FloatType>) ops.env().op("create.img").input(new FinalDimensions(sphericalInterval), new FloatType()).apply());

            RealRandomAccessible interpolatedMovingView = extendConstAndInterpolate(moving, mean);

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

        public MutableTriple<Integer, Double, Double> findHighestRotationCorrelation(RandomAccessibleInterval<FloatType> reference, RandomAccessibleInterval<FloatType> moving, HashMap<Integer, Img<FloatType>> toSearch) {
//            uiService.show("Moving in findHighestRotationCorrelation", datasetService.create(ImgView.wrap(moving)));

            for (Integer axis : toSearch.keySet()) {
                RandomAccessibleInterval rotatedReferenceView = null;
                RandomAccessibleInterval rotatedMovingView = null;

                switch (axis) {
                    case 0:
                        rotatedReferenceView = rotateImage(reference, mean, rotate0to2);
                        rotatedMovingView = rotateImage(moving, mean, rotate0to2);
                        break;
                    case 1:
                        rotatedReferenceView = rotateImage(reference, mean, rotate1to2);
                        rotatedMovingView = rotateImage(moving, mean, rotate1to2);
                        break;
                    case 2:
                        rotatedReferenceView = reference;
                        rotatedMovingView = moving;
                        break;
                }

                setSingleAxisCorrelationImage(toSearch.get(axis), rotatedReferenceView, rotatedMovingView);

//                uiService.show(axis + "-axis corr", Views.zeroMin(toSearch.get(axis)));
            }
            return findHighestRotationalPointAmong(toSearch);
        }


        /*
         * There are a few variables pertaining to the inclination value from the rotational cross-correlation result in
         * terms of the resulting correct rotation.
         *  1. Inclination does not tell us which of the other two non-azimuthal axes to rotate. In truth, it could be
         *  neither of these axes, but an arbitrary axis orthogonal to the azimuth axis. However, we simply test
         *  the two primary axes.
         *  2. We also do not know of the correct rotation order, as rotations are not commutative.
         *  Azimuth then Inclination, or Inclination then Azimuth?
         *  3. In spherical coordinates, inclination is non-negative, as it is simply the angle away from the pole.
         *  However, the optimal correlation angle can be negative. Thus, we must test both positive and negative values
         *  for the inclination.
         *
         * Each of the above parameters has two possiblities that we must test, for a total of 8 possible unique combinations.
         * This method tests each of them for which generates the highest correlation.
         */
        public List<ValuePair<Integer,Double>> getOptimalRotationOrder(RandomAccessibleInterval<FloatType> reference, RealTransformRealRandomAccessible<FloatType, AffineTransform3D> moving, MutableTriple<Integer, Double, Double> rotations){

            //If inclination angle is 0, simply return the Azimuth rotation
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

            //Brute force search each option
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

            //Each pair below tests both possible axes for inclination.
            //Azimuth first, then inclination
            fullList.add(getOrderedRotationList(rotations.getLeft(), rotations.getRight(), remaining.get(0), rotations.getMiddle()));
            fullList.add(getOrderedRotationList(rotations.getLeft(), rotations.getRight(), remaining.get(1), rotations.getMiddle()));

            //Inclination first, then Azimuth
            fullList.add(getOrderedRotationList(remaining.get(0), rotations.getMiddle(), rotations.getLeft(), rotations.getRight()));
            fullList.add(getOrderedRotationList(remaining.get(1), rotations.getMiddle(), rotations.getLeft(), rotations.getRight()));

            //Remaining tests are same as above, but with negative values for the inclinations.
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

        private RandomAccessibleInterval<FloatType> getDoubleRotatedImage(RealTransformRealRandomAccessible<FloatType, AffineTransform3D> image, Interval interval, List<ValuePair<Integer,Double>> orderedRotations){
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

        private float getDirectCorrelation(RandomAccessibleInterval<FloatType> reference, RandomAccessibleInterval<FloatType> moving){
            //Img<FloatType> multiplied = (Img<FloatType>)ops.env().op("create.img").input(reference, new FloatType()).apply();
            //SciJava ops "math.mul" doesn't work here.
            //ops.env().op("math.mul").input(reference, Views.interval(Views.extendZero(moving), reference)).output(multiplied).compute();


//            RandomAccessibleInterval<FloatType> tempref = (RandomAccessibleInterval<FloatType>) ops.env().op("copy.rai").input(reference).apply();
//            RandomAccessibleInterval<FloatType> tempmoving = (RandomAccessibleInterval<FloatType>) ops.env().op("copy.rai").input(moving).apply();
//            final float tempRefmean = ijOps.stats().mean(reference).getRealFloat();
//            LoopBuilder.setImages(tempref).forEachPixel((pixel)-> {pixel.setReal(pixel.get()- tempRefmean);});
//
//            final float tempMovemean = ijOps.stats().mean(moving).getRealFloat();
//            LoopBuilder.setImages(tempmoving).forEachPixel((pixel)-> {pixel.setReal(pixel.get()- tempMovemean);});

            IterableInterval multiplied = ijOps.math().multiply(reference,Views.interval(Views.extendZero(moving), reference));

            DoubleType result = (DoubleType) ops.env().op("stats.sum").input(multiplied).apply();

            return result.getRealFloat();
        }

        private RandomAccessibleInterval rotateImage(RandomAccessibleInterval<FloatType> input, double mean, AffineTransform3D transform) {
            return getCurrentWorkingIntervalView(
                    RealViews.transform(extendConstAndInterpolate(input, mean), transform),
                    input
            );
        }

        private void setSingleAxisCorrelationImage(Img<FloatType> correlationImage, RandomAccessibleInterval<FloatType> reference, RandomAccessibleInterval<FloatType> moving) {
            FFTConvolution fftConvolution;
            ExecutorService service = Executors.newCachedThreadPool();

//            uiService.show("Reference spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(reference)), 0, 2));
//            uiService.show("Moving spherical", Views.rotate(Views.zeroMin(convertToSphericalCoordinates(moving)), 0, 2));


            fftConvolution = new FFTConvolution(
                    new ExtendedRandomAccessibleInterval(convertToSphericalCoordinates(reference, mean), new OutOfBoundsSpherical3DFactory<>()),
                    correlationImage,
                    new ExtendedRandomAccessibleInterval(convertToSphericalCoordinates(moving, mean), new OutOfBoundsSpherical3DFactory<>()),
                    correlationImage,
                    (ImgFactory<ComplexFloatType>) ops.env().op("create.imgFactory").input(correlationImage, new ComplexFloatType()).apply(),
                    service);
            fftConvolution.setComputeComplexConjugate(true);
            fftConvolution.setOutput(correlationImage);
            fftConvolution.convolve();
        }

        private MutableTriple<Integer, Double, Double> findHighestRotationalPointAmong(HashMap<Integer, Img<FloatType>> toSearch) {
            //returns dimension, inclination, and azimuthal angle for rotation
            TreeMap<Double, MutableTriple<Integer, Double, Double>> maxCorrelations= new TreeMap<>();
            double inclineRadians = 0.0;
            double azimuthRadians = 0.0;

            for (Integer i : toSearch.keySet()) {
                Double max = Double.NEGATIVE_INFINITY;
                MutableTriple<Integer, Long, Long> maxPoint = new MutableTriple<>();

                Cursor<FloatType> cursor = toSearch.get(i).localizingCursor();

                while (cursor.hasNext()) {
                    if (cursor.next().get() > max) {
                        max = (double) cursor.get().get();
                        maxPoint = new MutableTriple<>(i, cursor.getLongPosition(1), cursor.getLongPosition(2));
                    }
                }
                inclineRadians = (maxPoint.getMiddle() - ((toSearch.get(maxPoint.getLeft()).dimension(1) - 1) / 2.0)) * (1/rotationalScaleFactor);
                azimuthRadians = (maxPoint.getRight() - ((toSearch.get(maxPoint.getLeft()).dimension(2) - 1) / 2.0)) * (1/rotationalScaleFactor);

                if(i == 0){
                    azimuthRadians = -azimuthRadians;
                }

                maxCorrelations.put(max, new MutableTriple<>(i, inclineRadians, azimuthRadians));
            }

            List<Double> keys = new ArrayList<>(maxCorrelations.keySet());

            for (double key : keys) {
                if (Math.abs(maxCorrelations.get(key).getMiddle()) < Double.MIN_VALUE && Math.abs(maxCorrelations.get(key).getRight()) < Double.MIN_VALUE) {
                    maxCorrelations.remove(key);
                }
            }

            if(maxCorrelations.isEmpty()){
                logService.info("No optimal rotational correlation found.");
                return new MutableTriple<>(0, 0.0, 0.0);
            }

            /*
            Left: Axis perspective
            Middle: Incline position
            Right: Azimithual position
             */

            logService.info("Brightest point in Rotational Correlation images, from perspective of axis " + maxCorrelations.lastEntry().getValue().getLeft() +
                    ":\n\tInclination: " + Math.toDegrees(maxCorrelations.lastEntry().getValue().getMiddle()) + "\n\tAzimuth: " + Math.toDegrees(maxCorrelations.lastEntry().getValue().getRight()));

            return maxCorrelations.lastEntry().getValue();
        }


        private Img<FloatType> convertToSphericalCoordinates(RandomAccessibleInterval<FloatType> input, double mean) {

            RealTransformRealRandomAccessible sphericalInput = RealViews.transform(
                    RealViews.transformReal(extendConstAndInterpolate(input, mean), SphericalToCartesianTransform3D.getInstance().inverse()),
                    new Scale3D(1, rotationalScaleFactor, rotationalScaleFactor)
            );

            RandomAccessibleInterval<FloatType> sphericalRAI =
                    Views.interval(
                        Views.raster(
                                sphericalInput
                        ),
                        sphericalInterval
                    );
            Img<FloatType> returnImg = (Img<FloatType>) ops.env().op("create.img").input(sphericalInterval, new FloatType()).apply();

            LoopBuilder.setImages(returnImg, sphericalRAI).multiThreaded().forEachPixel((r,s) ->{
                r.setReal(s.getRealFloat());
            });

            compensateForPoleRepeat(returnImg);

//            long[] min = sphericalInterval.minAsLongArray();
//            long[] max = sphericalInterval.maxAsLongArray();
//            min[1] = -max[1];
//            FinalInterval expanded = new FinalInterval(min, max);
//            uiService.show(

//            );

            return returnImg;
        }
    }

    /*
     * For spherical coordinate transform data, the Intensity values along the Pole axis (inclination 0 and 180), are
     * repeated across all azimuthal values. During cross-ccorrelation of this spherical coordinate data this has the
     * effect of enhancing the correlation for no inclination change, particularly if the polar axis happens to have a few
     * data points in alignment, even if the registration is actually off. There are a few options to correct for this
     * issue, though each has its own drawbacks:
     *  1. Instead of using spherical coordinates, we could switch to using cylindrical coordinates. This entirely
     *  removes the pole issue, and there would be no repeated data. However, this means that we'll only get a single
     *  rotational axis for cross-correlation, potentially increasing the likelihood of getting stuck in local minima.
     *  Additionally, the "scale" axis for cylindrical coordinate data isn't a global scale, but just around that axis.
     *  This could cause issues if we want to add scaling registration in the future.
     *
     *  2. We could not include the poles in the spherical coordinate data. This would remove the issue, but could be very
     *  challenging to implement, particularly if we allow custom values from users, and would probably require a new
     * OutOfBounds factory, as the "OutOfBoundsSpherical3D" implementation expects the poles to be included.
     *
     *  3. We can simply change the intensity of the data at the poles to compensate for the repeats across the azimuthal
     *  axis. This is one of the simplest method to implement, correcting for the increased correlation along the azimuthal axis,
     *  though it does still mean that the repeats will be evaluated across the entire azimuthal axis for any inclination
     *  rotations during cross-correlation, though the intensity from this contribution would be reduced. Still, due to
     *  its simplicity this is the method I first attempted.
     *
     *  4. We could change the intensity at the poles to 0. This way the poles would simply not contribute to the cross
     *  correlation result at all. However, for incline rotation transforms, this completely eliminates 4 rows of data,
     *  resulting in a very consistant lower value for incline rotations.
     */
    private void compensateForPoleRepeat(RandomAccessibleInterval<FloatType> input){
        final long inclinationSize = input.dimension(1);
        final float correctionFactor = input.dimension(2);

        Views.hyperSlice(input, 1, 0).cursor().forEachRemaining((pixel) ->{
            if(pixel.getRealFloat() > 0.0) {
                pixel.setReal(pixel.getRealFloat() / correctionFactor);
            }
        });

        Views.hyperSlice(input, 1, inclinationSize-1).cursor().forEachRemaining((pixel) ->{
            if(pixel.getRealFloat() > 0.0) {
                pixel.setReal(pixel.getRealFloat() / correctionFactor);
            }
        });
    }

    public class CorrelationTranslation {

        public double[] findMovingImageTranslation(RandomAccessibleInterval reference, RandomAccessibleInterval moving){
            CorrelationFunctions correlationFunctions = new CorrelationFunctions();
            RandomAccessibleInterval<FloatType> correlationImage = correlationFunctions.correlate(reference,moving);

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

        //todo: Get SciJava ops correlation working, then test speed compared to FFTConvolution
//        public RandomAccessibleInterval<FloatType> correlate(RandomAccessibleInterval input, RandomAccessibleInterval kernel){
//            Interval bounds = new FinalInterval(greatestOddDimensions(input, kernel));
//
//            return (RandomAccessibleInterval<FloatType>) ops.env().op("filter.correlate").input(
//                    input,
//                    kernel,
//                    new FloatType(),
//                    new ComplexFloatType())
//                    .apply();
//        }

        public Img<FloatType> correlate(RandomAccessibleInterval input, RandomAccessibleInterval kernel){
            FFTConvolution fftConvolution;
            ExecutorService service = Executors.newCachedThreadPool();
            Dimensions dims = greatestOddDimensions(input, kernel);

            Img<FloatType> crossCorrelation = (Img<FloatType>) ops.env().op("create.img").input(dims, new FloatType()).apply();

            Interval bounds = new FinalInterval(crossCorrelation);

            fftConvolution = new FFTConvolution(Views.extendZero(input), bounds,Views.extendZero(kernel), bounds, (ImgFactory<ComplexFloatType>) ops.env().op("create.imgFactory").input(bounds, new ComplexFloatType()).apply(), service);
            fftConvolution.setComputeComplexConjugate(true);
            fftConvolution.setOutput(crossCorrelation);
            fftConvolution.convolve();

//            uiService.show("Crosscorrelated image", Views.hyperSlice(crossCorrelation, 2, (crossCorrelation.dimension(2)-1)/2));

            return crossCorrelation;
        }

        private RandomAccessibleInterval expandImage(RandomAccessibleInterval input, Interval bounds){
            return Views.interval(
                    Views.expandZero(input),
                    bounds
            );
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
