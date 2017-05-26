---
author: J. L. Nabagło
title: "Optimising Experiments for Photometric Redshifts"
engine: xelatex
---

##What is your project?

I am working with photometric redshifts. Think of the problem of finding the redshift of an object. The redshift of the signal (radio, infrared, or visible light) coming towards us from an object is the way we measure how much the Doppler Effect affects said signal. The larger the redshift, the faster the object moves away from us.

Generally, due to the expansion of the universe, the larger the redshift, the further an object is away from us. Also, measuring the redshift of individual components of a galaxy lets us examine its dynamics, and answer questions such as, how much mass does it contain, and how fast is it spinning?

Traditionally, the redshift of an object was measured by pointing a spectroscope at it, comparing the hydrogen lines against the baseline, and working out the difference. The problem introduced is that a spectroscope must be directed at each individual object we wish to study. Since there are many objects in the observable universe, this is too many things.

Instead, we can take photometric measurements. These are akin to a photograph with a camera, since we capture a limited number of colours (in the case of a camera, red, green, and blue) instead of the full spectrum. From these, by applying machine learning techniques, we can find the redshift.

This has been done before, but a more interesting problem is: given a set objects for which we do have photometric measurements, but we do not have spectroscopic measurements, what is the best object to point a spectroscope at next?
