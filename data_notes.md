# Notes about SDSS data

Sloan Digital Sky Survey (SDSS):
[http://classic.sdss.org](http://classic.sdss.org)

Data Release DR7 (with lots of description):
[http://classic.sdss.org/dr7/](http://classic.sdss.org/dr7/)


## A subset of labelled SDSS data

We retrieve our data via SQL from the newest release page (DR12) even though the data we actually access is DR7 data that is contained within the newer larger set
SDSS SkyServer:
[http://skyserver.sdss.org/public/en/tools/search/sql.aspx](http://skyserver.sdss.org/public/en/tools/search/sql.aspx)


We will choose a sky area to limit the volume of data we deal with, and download tabular data with contents of two kinds:

- imaging data with photometry (flux measurements in five spectral passbands = "ugriz"): this data is easy to obtain and available for every object found in the whole survey, this is 23 columns in total

- spectroscopic data with a human expert classification and redshift "z" (= distance): this data is expensive to obtain and available only for a targetted subset of objects, it'll be just four columns

Here is the approporiate SQL query for a ~22 MB FITS file: (I use FITS tables a lot, but CSV etc. is available as well)

-- This query does a table JOIN between the imaging (PhotoObj) and spectra
-- (SpecObj) tables and includes the necessary columns in the SELECT to upload
-- the results to the SAS (Science Archive Server) for FITS file retrieval.

```
SELECT 
 p.ra,p.dec,p.psfMag_u,p.psfMag_g,p.psfMag_r,p.psfMag_i,p.psfMag_z,p.psfMagErr_u,p.psfMagErr_g,p.psfMagErr_r,p.psfMagErr_i,p.psfMagErr_z,p.petroMag_u,p.petroMag_g,p.petroMag_r,p.petroMag_i,p.petroMag_z,p.petroMagErr_u,p.petroMagErr_g,p.petroMagErr_r,p.petroMagErr_i,p.petroMagErr_z,p.petroRad_r,
 s.class, ISNULL(s.z,-1) as Redshift, ISNULL(s.zErr,0) as Red_Err
FROM PhotoObj AS p
   LEFT JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   p.psfMag_r < 21 AND ( p.type = 3 OR p.type = 6) AND p.ra between 202 and 207 and p.dec between 0 and 2
```

We restricted the sample to objects brighter than 21 mag in the r-band in order to remove plenty of noisy faint objects. The sky area we select is 5 degrees wide (in "ra" coordinate) and 2 degrees tall (in dec coordinate), totalling 10 square degrees. This is approx. twice the area that SkyMapper sees in every single shot, and 1 part in 2000 of the total sky area mapped with SkyMapper.

The columns are:

- **ra** - ra position in degrees
- **dec** - dec position in degrees
- **psfMag_u** - magnitude measurement in u-band assuming object is a point source
- **psfMagErr_u** - error of that measurement
- **petroMag_u** - magnitude measurement in u-band assuming object is an extended source
- **petroMagErr_u** - error of that measurement - same for griz bands
- **petroRad_r** - size measurement in r-band in arc seconds class
- **spectrosopic class**, expert opinion (null, STAR, GALAXY, QSO)
- **Redshift** - redshift (cosmological distance) of object, from spectrum, expert opinion
- **Red_Err** - error of redshift measurement

There should be 194,471 rows by 26 columns. Negative numbers indicate absent information (exception: dec, which is celestial latitude, can be negative although we chose to cut of at 0 deg).

Some issues: there are currently no columns specifying data quality on an object-by-object basis, although a vast and complex amount of such information exists in the database. It is not clear to me what default selection might have been applied.


## Reddening correction

go to the table schema web page, you see all the columns in the PhotoObj SQL table for the SDSS:   
[http://skyserver.sdss.org/dr12/en/help/browser/browser.aspx#&&history=description+PhotoObjAll+U](http://skyserver.sdss.org/dr12/en/help/browser/browser.aspx#&&history=description+PhotoObjAll+U)

Quite far down in the list there are five columns called extinction_* with * = u/g/r/i/z  so you can add these to your query.

These five columns specify values for the estimated absorption of light due to dust in units of magnitudes. We denote them in astronomy as “A_u”, “A_g”, etc. — these need to be subtracted from the ugriz filter magnitudes, such that the filter mag value gets smaller, corresponding to (in the astronomical system) a brighter true flux compared to the fainter dust-absorbed appearance. Obviously, if you already have formed colour indices from the magnitudes such as u-g, you can also express the dust extinction in terms of colour indices, which we then call “reddening”. The reddening in u-g due to dust is denoted in astronomy with “E_u-g = A_u - A_g”.

The version of this set of five A_band numbers (or four E_band1-band2) we can denote “SFD98”, and it is one specific estimate of the dust extinction long held to be the default choice. Using this should already show ‘some' improvement in distinguishing the classes.

Then we will want to investigate variations of this version and see whether the second-order effect of further improvement is visible or not. Additional versions are explored in Wolf (2014), pdf attached below. These additional versions all have one free parameter because the analysis trying to refine our estimate of dust extinction constrain not the A_band's directly, but they constrain the E_b1-b2’s instead.

So, version 2, called “SF11” is obtained by calculating a reference reddening and then applying a new “extinction curve” to get all the A_band’s:

Calculate first a reference value of   E_B-V = A_r/2.751   from the A_r value in the SFD98 estimate retrieved from the SDSS database
Calculate next the SF11 estimates using
  A_u = E_B-V * 4.239
  A_g = E_B-V * 3.303
  A_r = E_B-V * 2.285
  A_i = E_B-V * 1.698
  A_z = E_B-V * 1.263

(just for completeness, you should be able to recover the SFD98 A_band’s from your newly obtained E_B-V using
  A_u = E_B-V * 5.155
  A_g = E_B-V * 3.793
  A_r = E_B-V * 2.751
  A_i = E_B-V * 2.086
  A_z = E_B-V * 1.479
this was version 1)


Version 3 we might call W14 for now. It is a tad more complicated. We assume we already have E_B-V calculated in the previous step. Now:

Remap E_B-V onto a new value using     

If E_B-V in [0,0.04]  Ecorr_B-V = E_B-V
If E_B-V in [0.04,0.08]     Ecorr_B-V = E_B-V + 0.5 * (E_B-V - 0.04)
If E_B-V in [0.08,+inf]
Ecorr_B-V = E_B-V + 0.02

Calculate next the W14 estimates using
  A_u = Ecorr_B-V * 4.305
  A_g = Ecorr_B-V * 3.288
  A_r = Ecorr_B-V * 2.261
  A_i = Ecorr_B-V * 1.714
  A_z = Ecorr_B-V * 1.263

To be honest, the main change introduced by W14 is the remapping of the E_B-V scale, not so much the change in the A_band factors. The latter alone should have little effect, but the former would have a bit more effect.
