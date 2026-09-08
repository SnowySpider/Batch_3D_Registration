import net.imglib2.Interval;
import net.imglib2.util.Intervals;

public class B3dParameters {
    private double[] subsampleScale;
    private double rotationalScaleFactor;
    private boolean translateByCorrelate;

    public static enum Resolutions{
        VeryLow, Low, Moderate, Normal, High, VeryHigh, Native
    }

    public B3dParameters(Interval interval, Resolutions resolution){
        switch (resolution) {
            case VeryLow:
                this.setParameters(interval, 100000, 10.0, false);
                break;
            case Low:
                this.setParameters(interval, 200000, 5.0, false);
                break;
            case Moderate:
                this.setParameters(interval, 350000, 3.0, false);
                break;
            case Normal:
                this.setParameters(interval, 500000, 2.0, true);
                break;
            case High:
                this.setParameters(interval, 1000000, 1.0, true);
                break;
            case VeryHigh:
                this.setParameters(interval, 10000000, 0.5, true);
                break;
            case Native:
                this.setParameters(interval, Intervals.numElements(interval), 0.1, true);
        }
    }

    public B3dParameters(Interval interval, long targetPixels, double degreeAccuracy, boolean translateByCorrelate){
        this.setParameters(interval, targetPixels, degreeAccuracy, translateByCorrelate);
    }

    public void setParameters(Interval interval, long targetPixels, double degreeAccuracy, boolean translateByCorrelate){
        double scaleFactor = (double) targetPixels/Intervals.numElements(interval);
        scaleFactor = Math.cbrt(scaleFactor);
        if (scaleFactor > 1)
                scaleFactor = 1;
        this.subsampleScale = new double[]{scaleFactor,scaleFactor,scaleFactor};

        this.rotationalScaleFactor = (180/degreeAccuracy)/Math.PI;

        this.translateByCorrelate = translateByCorrelate;
    }

    public double[] getSubsampleScale(){
        return subsampleScale.clone();
    }

    public double getRotationalScaleFactor(){
        return rotationalScaleFactor;
    }

    public boolean getTranslateByCorrelate(){
        return translateByCorrelate;
    }


}
