Physics Analysis Background
============

:math:`t\bar{t}` Events
-----------------
The Analysis Grand Challenge aims to demonstrate the integration of different software components in a realistic physics analysis. The task chosen for CMS open data is a measurement of the :math:`t\bar{t}` (top-antitop quark pair production) crosssection  measurement using events with at least four jets and one lepton in the final state. This signature has two possibilities, depending on whether the lepton/neutrino pair originates from the top or antitop quark:

.. image:: images/ttbar.png
  :width: 80%
  :alt: Diagram of a :math:`t\bar{t}` event with four jets and one lepton.

The jets are illustrated in purple. These are collimated showers of hadrons resulting from quarks, which cannot exist on their own due to colour confinement.

Here is an example of the above signature in CMS Open Data. Since this is Monte Carlo simulated data, we can look at the particles that the jets originate from. The jets are plotted as circles in the :math:`\eta`-:math:`\phi` plane, and color-coordinated with the truth particles they are matched to. Note that :math:`\phi` is a circular variable, so the top of the graph matches to the bottom. The jets that are outlined in solid black are b-tagged, which means that they have properties which strongly indicate that they originate from bottom quarks.

.. image:: images/event3.png
  :width: 80%
  :alt: Example of a :math:`t\bar{t}` event in our signal region.
  
The tree of the above event looks something like::

    g
    ├── t
    │   ├── W+
    │   │   ├── mu+
    │   │   └── nu(mu)
    │   └── b
    └── t~
        ├── W-
        │   ├── s
        │   └── c~
        └── b~
        
        
Machine Learning Component
-----------------