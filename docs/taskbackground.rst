Physics Analysis Background
===============================================================

:math:`t\bar{t}` Events
---------------------------------------------------------------
The Analysis Grand Challenge aims to demonstrate the integration of different software components in a realistic physics analysis. The task chosen for CMS open data is a measurement of the :math:`t\bar{t}` (top-antitop quark pair production) crosssection  measurement using events with at least four jets and one lepton in the final state. This signature has two possibilities, depending on whether the lepton/neutrino pair originates from the top or antitop quark:

.. image:: images/ttbar.png
  :width: 80%
  :alt: Diagram of a :math:`t\bar{t}` event with four jets and one lepton.

The jets are illustrated in purple. These are collimated showers of hadrons resulting from quarks, which cannot exist on their own due to colour confinement.

Here is an example of the above signature in CMS Open Data. Since this is Monte Carlo simulated data, we can look at the particles that the jets originate from. The jets are plotted as circles in the :math:`\eta`-:math:`\phi` plane, and color-coordinated with the truth particles they are matched to. Note that :math:`\phi` is a circular variable, so the top of the graph matches to the bottom. The jets that are outlined in solid black are b-tagged, which means that they have properties which strongly indicate that they originate from bottom quarks.

To look at more events, take a look at **Plot :math:`t\bar{t}` Events**.

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
        
Top Mass Reconstruction
---------------------------------------------------------------
To measure the :math:`t\bar{t}` crosssection, we use an observable that approximately reconstructs the top quark mass. We do this in two different ways. The first uses no machine learning. Within an event, the trijet system with the highest transverse momentum (:math:`p_T`) is selected. We then calculate the combined mass of these three jets.

Machine Learning Component
---------------------------------------------------------------
Most modern high energy physics analyses use some form of machine learning (ML), so a machine learning task has been incorporated into the AGC :math:`t\bar{t}` crosssection  measurement to reflect this development. The method described above to reconstruct the top mass attempts to correctly select all three jets on the hadronic side of the collision. Using ML, we can go beyond this task by attempting to correctly assign each jet with its parent parton. This should allow for a more accurate top mass reconstruction as well as access to new observables, such as the angle between the jet on the leptonic side of the collision and the lepton, or the angle between the two W jets.

The strategy used for this jet-parton assignment task is as follows:

In each event, we want to associate four jets to three labels. We want to label two jets as :math:`W` (considering these two to be indistinguishable), one jet as :math:`top_{hadron}` (the top jet on the side of hadronic decay), and one as :math:`top_{lepton}` (the top jet on the side of leptonic decay). This is visualized in the diagram below:

.. image:: images/ttbar_labels.png
  :width: 80%
  :alt: Diagram of a :math:`t\bar{t}` event with the three machine learning labels for jets.
  
In each event, we consider each permutation of jets assigned to these labels, restricting to the leading :math:`N` jets. The number of such permutations (assuming the event has at least :math:`N` jets and that :math:`N\geq 4`) is :math:`N!/(2\cdot (N-4)!)`. For example, if there are 4 jets in an event, we consider :math:`4!/2=12` permutations. The :math:`4!` comes from labelling 4 jets, while dividing by 2 accounts for the fact that two of the jets are labelled indistinguishably. If there are more than 4 jets, the remainder are assigned to a fourth category, "other". Jets are also assigned to this category indistinguishably. For example if :math:`N=7` and we have 7 jets in an event, we consider :math:`7!/(2\cdot 3!)=420` permutations, since we assign labels to 7 jets. We assign 2 jets to "W" indistinguishably, then the three remainder to "other" indistinguishably.

To vizualize the :math:`N=4` scenario, view the diagram below:

.. image:: images/permutations.png
  :width: 80%
  :alt: Possible jet-label assignments for :math:`N=4` scenario.
  
