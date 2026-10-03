package outofboundsspherical;

import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.outofbounds.OutOfBoundsFactory;

/**
 * Factory for {@link OutOfBoundsSpherical3D}, for use with
 * {@code Views.extend(source, factory)}.
 *
 * @param <T>
 * @param <F>
 */
public class OutOfBoundsSpherical3DFactory< T, F extends Interval & RandomAccessible< T > >
        implements OutOfBoundsFactory< T, F >
{
    private final int rDim;

    private final int thetaDim;

    private final int phiDim;

    private final boolean skipBoundary;

    private final boolean radiusStartsAtOrigin;

    /** Default layout: (r, theta, phi) = (0, 1, 2), skip boundary, radius from origin. */
    public OutOfBoundsSpherical3DFactory()
    {
        this( 0, 1, 2, true, true );
    }

    public OutOfBoundsSpherical3DFactory(
            final int rDim,
            final int thetaDim,
            final int phiDim,
            final boolean skipBoundary,
            final boolean radiusStartsAtOrigin )
    {
        this.rDim = rDim;
        this.thetaDim = thetaDim;
        this.phiDim = phiDim;
        this.skipBoundary = skipBoundary;
        this.radiusStartsAtOrigin = radiusStartsAtOrigin;
    }

    @Override
    public OutOfBoundsSpherical3D< T > create( final F f )
    {
        return new OutOfBoundsSpherical3D< T >( f, rDim, thetaDim, phiDim, skipBoundary, radiusStartsAtOrigin );
    }
}