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
