/*
 * #%L
 * ImgLib2: a general-purpose, multidimensional image processing library.
 * %%
 * Copyright (C) 2009 - 2025 Tobias Pietzsch, Stephan Preibisch, Stephan Saalfeld,
 * John Bogovic, Albert Cardona, Barry DeZonia, Christian Dietz, Jan Funke,
 * Aivar Grislis, Jonathan Hale, Grant Harris, Stefan Helfrich, Mark Hiner,
 * Martin Horn, Steffen Jaensch, Lee Kamentsky, Larry Lindsey, Melissa Linkert,
 * Mark Longair, Brian Northan, Nick Perry, Curtis Rueden, Johannes Schindelin,
 * Jean-Yves Tinevez and Michael Zinsmaier.
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package outofboundsspherical;

import net.imglib2.AbstractLocalizable;
import net.imglib2.Interval;
import net.imglib2.Localizable;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.outofbounds.OutOfBounds;
import net.imglib2.util.Util;

/**
 * Out-of-bounds strategy for a volume sampled on a regular grid in spherical
 * coordinates (r, &theta;, &phi;), e.g. the result of rendering a Cartesian
 * volume through
 * {@code net.imglib2.realtransform.SphericalToCartesianTransform3D.inverse()}.
 * <p>
 * Assumed grid layout (dimension indices are configurable, default
 * r = 0, &theta; = 1, &phi; = 2):
 * </p>
 * <ul>
 * <li>the <b>polar axis</b> (&theta;) spans pole to pole (inclination
 * 0..&pi; or elevation -&pi;/2..&pi;/2; the wrapping is identical in grid
 * coordinates),</li>
 * <li>the <b>azimuth axis</b> (&phi;) spans one full revolution,</li>
 * <li>the <b>radius axis</b> (r) starts at the origin if
 * {@code radiusStartsAtOrigin} is set.</li>
 * </ul>
 * <p>
 * Wrapping rules (let {@code s = 1} if {@code skipBoundary}, else {@code 0}):
 * </p>
 * <ul>
 * <li><b>Azimuth:</b> periodic with period {@code dimension - s}. With
 * {@code skipBoundary} the first and last sample are the same angle (e.g. 0 and
 * 2&pi;), so {@code max + 1} maps to {@code min + 1}.</li>
 * <li><b>Polar angle:</b> reflects at the poles, and the azimuth is rotated by
 * half a revolution (you leave the pole on the opposite meridian). With
 * {@code skipBoundary} the pole sample is on the grid and is not repeated
 * ({@code min - 1} maps to {@code min + 1}); otherwise the edge sample is
 * repeated ({@code min - 1} maps to {@code min}).</li>
 * <li><b>Radius &gt; max:</b> clamped to the outermost shell (an
 * {@link OutOfBounds} cannot synthesise a constant without being given
 * one).</li>
 * <li><b>Radius &lt; min:</b> if {@code radiusStartsAtOrigin}, a negative
 * radius is the antipodal point, (r, &theta;, &phi;) &rarr;
 * (-r, &pi; - &theta;, &phi; + &pi;), so the polar index is mirrored and the
 * azimuth rotated by half a revolution. Otherwise clamped to the innermost
 * shell.</li>
 * <li><b>Any further dimensions</b> (n &gt; 3) are clamped.</li>
 * </ul>
 * <p>
 * Positions inside the interval always read the real sample. A pixel-exact
 * half-turn requires the azimuth period ({@code dimension - s}) to be even;
 * otherwise the shift is rounded down.
 * </p>
 *
 * @param <T>
 *
 * @author Stephan Saalfeld (original periodic implementation)
 */
public class OutOfBoundsSpherical3D< T > extends AbstractLocalizable implements OutOfBounds< T >
{
    final protected RandomAccess< T > outOfBoundsRandomAccess;

    /** Dimensions of the wrapped {@link RandomAccessible}. */
    final protected long[] dimension;

    /** Minimum of the wrapped {@link RandomAccessible}. */
    final protected long[] min;

    /** Maximum of the wrapped {@link RandomAccessible}. */
    final protected long[] max;

    final protected boolean[] dimIsOutOfBounds;

    protected boolean isOutOfBounds = false;

    /** Index of the radius dimension. */
    final protected int rDim;

    /** Index of the polar angle (&theta;) dimension. */
    final protected int thetaDim;

    /** Index of the azimuth (&phi;) dimension. */
    final protected int phiDim;

    /** Whether first and last sample along the angular axes coincide. */
    final protected boolean skipBoundary;

    /** Whether the first radius sample is r = 0. */
    final protected boolean radiusStartsAtOrigin;

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
        rDim = outOfBounds.rDim;
        thetaDim = outOfBounds.thetaDim;
        phiDim = outOfBounds.phiDim;
        skipBoundary = outOfBounds.skipBoundary;
        radiusStartsAtOrigin = outOfBounds.radiusStartsAtOrigin;

        outOfBoundsRandomAccess = outOfBounds.outOfBoundsRandomAccess.copy();
    }

    /**
     * Default layout: dimensions (r, &theta;, &phi;) = (0, 1, 2), boundary
     * samples skipped, radius starting at the origin.
     */
    public < F extends Interval & RandomAccessible< T > > OutOfBoundsSpherical3D( final F f )
    {
        this( f, 0, 1, 2, true, true );
    }

    public < F extends Interval & RandomAccessible< T > > OutOfBoundsSpherical3D(
            final F f,
            final int rDim,
            final int thetaDim,
            final int phiDim,
            final boolean skipBoundary,
            final boolean radiusStartsAtOrigin )
    {
        super( f.numDimensions() );
        if ( n < 3 )
            throw new IllegalArgumentException( "Spherical wrapping needs at least 3 dimensions, got " + n );
        checkDim( rDim );
        checkDim( thetaDim );
        checkDim( phiDim );
        if ( rDim == thetaDim || rDim == phiDim || thetaDim == phiDim )
            throw new IllegalArgumentException( "r, theta and phi dimensions must be distinct" );

        this.rDim = rDim;
        this.thetaDim = thetaDim;
        this.phiDim = phiDim;
        this.skipBoundary = skipBoundary;
        this.radiusStartsAtOrigin = radiusStartsAtOrigin;

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
     * Maps the logical position (r, &theta;, &phi;, ...) to the in-bounds
     * position of the wrapped image and moves the underlying accessor there.
     * The three spherical axes are coupled (antipodal reflection and pole
     * crossings rotate the azimuth), so they are always mapped together.
     */
    private void updateAccess()
    {
        final int s = skipBoundary ? 1 : 0;

        long r = position[ rDim ];
        long t = position[ thetaDim ];
        long p = position[ phiDim ];

        final long minR = min[ rDim ];
        final long minT = min[ thetaDim ];
        final long minP = min[ phiDim ];
        final long nPhi = dimension[ phiDim ] - s; // samples per revolution
        final long hTheta = dimension[ thetaDim ] - s; // pole to pole, in samples

        long phiShift = 0;

        /* radius: negative radius is the antipodal point */
        if ( r < minR && radiusStartsAtOrigin )
        {
            r = 2 * minR - r;
            t = minT + ( dimension[ thetaDim ] - 1 ) - ( t - minT );
            phiShift += nPhi / 2;
        }
        r = clamp( r, rDim );

        /* polar angle: reflect at the poles, rotate azimuth by a half turn */
        if ( t < minT || t > max[ thetaDim ] )
        {
            if ( hTheta > 0 )
            {
                final long u = t - minT;
                final long crossings = Math.floorDiv( u, hTheta );
                final long rem = u - crossings * hTheta; // in [0, hTheta)
                if ( ( crossings & 1 ) != 0 )
                {
                    t = minT + hTheta - 1 + s - rem;
                    phiShift += nPhi / 2;
                }
                else
                    t = minT + rem;
            }
            else
                t = minT;
        }

        /* azimuth: periodic */
        if ( phiShift != 0 || p < minP || p > max[ phiDim ] )
        {
            /* skip boundary on */
            if ( nPhi > 0 )
                p = minP + Math.floorMod( p - minP + phiShift, nPhi );
            /* no boundary skip */
            else
                p = minP;
        }

        outOfBoundsRandomAccess.setPosition( r, rDim );
        outOfBoundsRandomAccess.setPosition( t, thetaDim );
        outOfBoundsRandomAccess.setPosition( p, phiDim );
        for ( int d = 0; d < n; ++d )
        {
            if ( d != rDim && d != thetaDim && d != phiDim )
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
        return outOfBoundsRandomAccess.get();
    }

    @Override
    public T getType()
    {
        return outOfBoundsRandomAccess.getType();
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