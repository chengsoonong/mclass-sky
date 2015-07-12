Photometric Data
================

In astronomy, there are two ways we can measure an object in the sky.
One method that is expensive method is to construct a spectrum. A
cheaper method is to take photometric measurements (i.e. putting
the light into some number of bins).


Dust Extinction
------------------------------
(notes from Chris)

The SDSS dataset contains five columns, which specify 
the estimated absorption of light due to dust in units of magnitudes. We denote them in astronomy as “A_u”, “A_g”, etc. — these need to be subtracted from the ugriz filter magnitudes, such that the filter mag value gets smaller, corresponding to (in the astronomical system) a brighter true flux compared to the fainter dust-absorbed appearance. Obviously, if you already have formed colour indices from the magnitudes such as u-g, you can also express the dust extinction in terms of colour indices, which we then call “reddening”. The reddening in u-g due to dust is denoted in astronomy with “E_u-g = A_u - A_g”.

The version of this set of five A_band numbers (or four E_band1-band2) we can denote “SFD98”, and it is one specific estimate of the dust extinction long held to be the default choice. Using this should already show ‘some' improvement in distinguishing the classes.

Then we will want to investigate variations of this version and see whether the second-order effect of further improvement is visible or not. Additional versions are explored in Wolf (2014), pdf attached below. These additional versions all have one free parameter because the analysis trying to refine our estimate of dust extinction constrain not the A_band's directly, but they constrain the E_b1-b2’s instead.

So, version 2, called “SF11” is obtained by calculating a reference reddening and then applying a new “extinction curve” to get all the A_band’s:

Calculate first a reference value of   E_{B-V} = A_r/2.751   from the A_r value in the SFD98 estimate retrieved from the SDSS database
Calculate next the SF11 estimates using

.. math::
    A_u = E_{B-V} * 4.239 \\
    A_g = E_{B-V} * 3.303 \\
    A_r = E_{B-V} * 2.285 \\
    A_i = E_{B-V} * 1.698 \\
    A_z = E_{B-V} * 1.263

(just for completeness, you should be able to recover the SFD98 A_band’s from your newly obtained E_{B-V} using

.. math::
    A_u = E_{B-V} * 5.155 \\
    A_g = E_{B-V} * 3.793 \\
    A_r = E_{B-V} * 2.751 \\
    A_i = E_{B-V} * 2.086 \\
    A_z = E_{B-V} * 1.479

this was version 1)

Version 3 we might call W14 for now. It is a tad more complicated. We assume we already have E_{B-V} calculated in the previous step. Now:

Remap E_{B-V} onto a new value using     

.. math::
    E_{B-V} \in [0,0.04]
        &\rightarrow Ecorr_{B-V} = E_{B-V} \\
    E_{B-V} \in [0.04,0.08]
        &\rightarrow   Ecorr_{B-V} = E_{B-V} + 0.5 * (E_{B-V} - 0.04) \\
    E_{B-V} \in [0.08,+\infty]
        &\rightarrow Ecorr_{B-V} = E_{B-V} + 0.02

Calculate next the W14 estimates using

.. math::
    A_u = Ecorr_{B-V} * 4.305 \\
    A_g = Ecorr_{B-V} * 3.288 \\
    A_r = Ecorr_{B-V} * 2.261 \\
    A_i = Ecorr_{B-V} * 1.714 \\
    A_z = Ecorr_{B-V} * 1.263

To be honest, the main change introduced by W14 is the remapping of the E_{B-V} scale, not so much the change in the A_band factors. The latter alone should have little effect, but the former would have a bit more effect.


