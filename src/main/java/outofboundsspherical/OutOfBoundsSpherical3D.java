/*
 * (keep your original license header here)
 */

package outofboundsspherical;

import net.imglib2.*;
import net.imglib2.outofbounds.OutOfBounds;
import net.imglib2.type.Type;
import net.imglib2.util.Util;

/**
 * Out-of-bounds strategy for a volume sampled on a regular grid in spherical
 * coordinates (r, &theta;, &phi;).
 * <p>
 * Assumed grid layout (dimension indices are configurable, default
 * r = 0, &theta; = 1, &phi; = 2):
 * </p>
 * <ul>
 * <li>the <b>azimuth axis</b> (&phi;) spans one full revolution and is
 * <b>periodic</b>. With {@code skipBoundary} the first and last sample are the
 * same angle (e.g. 0 and 2&pi;), so the period is {@code dimension - 1} and
 * {@code max + 1} maps to {@code min + 1}; otherwise the period is
 * {@code dimension}.</li>
 * <li>the <b>radius</b> (r) and <b>polar</b> (&theta;) axes return a
 * <b>constant value</b> whenever the position is outside the interval in
 * either of them.</li>
 * <li><b>Any further dimensions</b> (n &gt; 3) are clamped.</li>
 * </ul>
 * <p>
 * Positions inside the interval always read the real sample.
 * </p>
 *
 * @param <T>
 *
 * @author Andrew McCall
 */
public class OutOfBoundsSpherical3D< T extends Type< T > > extends AbstractLocalizable implements OutOfBounds< T >
{
    final protected RandomAccess< T > outOfBoundsRandomAccess;

    /** Value returned when r or theta is out of bounds. */
    final protected T value;

    /** Dimensions of the wrapped {@link RandomAccessible}. */
    final protected long[] dimension;

    /** Minimum of the wrapped {@link RandomAccessible}. */
    final protected long[] min;

    /** Maximum of the wrapped {@link RandomAccessible}. */
    final protected long[] max;

    final protected boolean[] dimIsOutOfBounds;

    /** True if any dimension is out of bounds (including wrapped phi). */
    protected boolean isOutOfBounds = false;

    /** True if r or theta is out of bounds, i.e. the constant is returned. */
    protected boolean isConstant = false;

    /** Index of the radius dimension. */
    final protected int rDim;

    /** Index of the polar angle (&theta;) dimension. */
    final protected int thetaDim;

    /** Index of the azimuth (&phi;) dimension. */
    final protected int phiDim;

    /** Whether first and last sample along the azimuth axis coincide. */
    final protected boolean skipBoundary;

    public OutOfBoundsSpherical3D( final OutOfBoundsSpherical3D< T > outOfBounds )
    {
        super( outOfBounds.numDimensions() );
        dimension = new long[ n ];
        min = new long[ n ];
        max = new long[ n ];
        dimIsOutOfBounds = new boolean[ n ];
        for ( int d = 0; d < n; ++d )
        {
            dimension[ d ] = outOfBounds.dimension[ d ];
            min[ d ] = outOfBounds.min[ d ];
            max[ d ] = outOfBounds.max[ d ];
            position[ d ] = outOfBounds.position[ d ];
            dimIsOutOfBounds[ d ] = outOfBounds.dimIsOutOfBounds[ d ];
        }
        isOutOfBounds = outOfBounds.isOutOfBounds;
        isConstant = outOfBounds.isConstant;
        rDim = outOfBounds.rDim;
        thetaDim = outOfBounds.thetaDim;
        phiDim = outOfBounds.phiDim;
        skipBoundary = outOfBounds.skipBoundary;
        value = outOfBounds.value.copy();

        outOfBoundsRandomAccess = outOfBounds.outOfBoundsRandomAccess.copy();
    }

    /**
     * Default layout: dimensions (r, &theta;, &phi;) = (0, 1, 2), boundary
     * samples skipped along &phi;.
     */
    public < F extends Interval & RandomAccessible< T > > OutOfBoundsSpherical3D( final F f, final T value )
    {
        this( f, value, 0, 1, 2, true );
    }

    public < F extends Interval & RandomAccessible< T > > OutOfBoundsSpherical3D(
            final F f,
            final T value,
            final int rDim,
            final int thetaDim,
            final int phiDim,
            final boolean skipBoundary )
    {
        super( f.numDimensions() );
        if ( n < 3 )
            throw new IllegalArgumentException( "Spherical wrapping needs at least 3 dimensions, got " + n );
        checkDim( rDim );
        checkDim( thetaDim );
        checkDim( phiDim );
        if ( rDim == thetaDim || rDim == phiDim || thetaDim == phiDim )
            throw new IllegalArgumentException( "r, theta and phi dimensions must be distinct" );

        this.value = value.copy();
        this.rDim = rDim;
        this.thetaDim = thetaDim;
        this.phiDim = phiDim;
        this.skipBoundary = skipBoundary;

        dimension = new long[ n ];
        f.dimensions( dimension );
        min = new long[ n ];
        f.min( min );
        max = new long[ n ];
        f.max( max );
        dimIsOutOfBounds = new boolean[ n ];

        outOfBoundsRandomAccess = f.randomAccess();
    }

    private void checkDim( final int d )
    {
        if ( d < 0 || d >= n )
            throw new IllegalArgumentException( "Dimension index out of range: " + d );
    }

    final protected void checkOutOfBounds()
    {
        isConstant = dimIsOutOfBounds[ rDim ] || dimIsOutOfBounds[ thetaDim ];
        for ( int d = 0; d < n; ++d )
        {
            if ( dimIsOutOfBounds[ d ] )
            {
                isOutOfBounds = true;
                return;
            }
        }
        isOutOfBounds = false;
    }

    /* Wrapping logic */

    private long clamp( final long p, final int d )
    {
        return Math.max( min[ d ], Math.min( max[ d ], p ) );
    }

    /**
     * Moves the underlying accessor to a valid in-bounds position. Phi is
     * wrapped periodically; all other dimensions are clamped (for r and theta
     * the clamped sample is never read, as the constant is returned instead).
     */
    private void updateAccess()
    {
        final int s = skipBoundary ? 1 : 0;
        final long nPhi = dimension[ phiDim ] - s; // samples per revolution

        long p = position[ phiDim ];
        if ( p < min[ phiDim ] || p > max[ phiDim ] )
        {
            if ( nPhi > 0 )
                p = min[ phiDim ] + Math.floorMod( p - min[ phiDim ], nPhi );
            else
                p = min[ phiDim ];
        }
        outOfBoundsRandomAccess.setPosition( p, phiDim );

        for ( int d = 0; d < n; ++d )
        {
            if ( d != phiDim )
                outOfBoundsRandomAccess.setPosition( clamp( position[ d ], d ), d );
        }
    }

    /** Recomputes flags and the underlying accessor for all dimensions. */
    private void refreshAll()
    {
        for ( int d = 0; d < n; ++d )
            dimIsOutOfBounds[ d ] = position[ d ] < min[ d ] || position[ d ] > max[ d ];
        checkOutOfBounds();
        updateAccess();
    }

    /* OutOfBounds */

    @Override
    public boolean isOutOfBounds()
    {
        return isOutOfBounds;
    }

    /* Sampler */

    @Override
    public T get()
    {
        return isConstant ? value : outOfBoundsRandomAccess.get();
    }

    @Override
    public T getType()
    {
        return value;
    }

    @Override
    final public OutOfBoundsSpherical3D< T > copy()
    {
        return new OutOfBoundsSpherical3D< T >( this );
    }

    /* Positionable */

    @Override
    final public void fwd( final int d )
    {
        setPosition( position[ d ] + 1, d );
    }

    @Override
    final public void bck( final int d )
    {
        setPosition( position[ d ] - 1, d );
    }

    @Override
    final public void setPosition( final long position, final int d )
    {
        final boolean wasOutOfBounds = isOutOfBounds;
        this.position[ d ] = position;
        dimIsOutOfBounds[ d ] = position < min[ d ] || position > max[ d ];
        checkOutOfBounds();

        if ( !wasOutOfBounds && !isOutOfBounds )
            // fast path: underlying position equals logical position
            outOfBoundsRandomAccess.setPosition( position, d );
        else
            updateAccess();
    }

    @Override
    public void move( final long distance, final int d )
    {
        setPosition( position[ d ] + distance, d );
    }

    @Override
    public void move( final int distance, final int d )
    {
        move( ( long ) distance, d );
    }

    @Override
    public void move( final Localizable localizable )
    {
        for ( int d = 0; d < n; ++d )
            position[ d ] += localizable.getLongPosition( d );
        refreshAll();
    }

    @Override
    public void move( final int[] distance )
    {
        for ( int d = 0; d < n; ++d )
            position[ d ] += distance[ d ];
        refreshAll();
    }

    @Override
    public void move( final long[] distance )
    {
        for ( int d = 0; d < n; ++d )
            position[ d ] += distance[ d ];
        refreshAll();
    }

    @Override
    public void setPosition( final int position, final int d )
    {
        setPosition( ( long ) position, d );
    }

    @Override
    public void setPosition( final Localizable localizable )
    {
        for ( int d = 0; d < n; ++d )
            position[ d ] = localizable.getLongPosition( d );
        refreshAll();
    }

    @Override
    public void setPosition( final int[] position )
    {
        for ( int d = 0; d < n; ++d )
            this.position[ d ] = position[ d ];
        refreshAll();
    }

    @Override
    public void setPosition( final long[] position )
    {
        for ( int d = 0; d < n; ++d )
            this.position[ d ] = position[ d ];
        refreshAll();
    }

    /* Object */

    @Override
    public String toString()
    {
        return Util.printCoordinates( position ) + " = " + get();
    }
}