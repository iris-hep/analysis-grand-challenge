Physics Analysis Background
===============================================================

:math:`t\bar{t}` Events
---------------------------------------------------------------
Implementations of analysis tasks in the Analysis Grand Challenge aim to demonstrate the integration of different software components in a realistic physics analysis. The task chosen for CMS open data is a measurement of the :math:`t\bar{t}` (top-antitop quark pair production) cross-section  measurement using events with at least four jets and one lepton in the final state. This signature has two possibilities, depending on whether the lepton/neutrino pair originates from the top or antitop quark:

.. image:: images/ttbar.png
  :width: 80%
  :alt: Diagram of a :math:`t\bar{t}` event with four jets and one lepton.

The jets are illustrated in purple. These are collimated showers of hadrons resulting from quarks, which cannot exist on their own due to colour confinement.

Here is an example of the above signature in CMS Open Data. Since this is Monte Carlo simulated data, we can look at the particles that the jets originate from. The jets are plotted as circles in the :math:`\eta`-:math:`\phi` plane, and color-coordinated with the truth particles they are matched to. Note that :math:`\phi` is a circular variable, so the top of the graph matches to the bottom. The jets that are outlined in solid black are b-tagged, which means that they have properties which strongly indicate that they originate from bottom quarks.

To look at more events, take a look at **Plot :math:`t\bar{t}` Events**.

.. image:: images/event3.png
  :width: 80%
  :alt: Example of a :math:`t\bar{t}` event in our signal region.
  
The tree of the above event looks something like

.. code-block:: text

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
To measure the :math:`t\bar{t}` cross-section, we use an observable that approximately reconstructs the top quark mass. We do this in two different ways. The first uses no machine learning. Within an event, the trijet system with the highest transverse momentum (:math:`p_T`) is selected. We then calculate the combined mass of these three jets.

Machine Learning Component
---------------------------------------------------------------
Most modern high energy physics analyses use some form of machine learning (ML), so a machine learning task has been incorporated into the AGC :math:`t\bar{t}` cross-section  measurement to reflect this development. The method described above to reconstruct the top mass attempts to correctly select all three jets on the hadronic side of the collision. Using ML, we can go beyond this task by attempting to correctly assign each jet with its parent parton. This should allow for a more accurate top mass reconstruction as well as access to new observables, such as the angle between the jet on the leptonic side of the collision and the lepton, or the angle between the two W jets.

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
  
For each permutation, we calculate 20 features:

#. :math:`\Delta R` between the :math:`top_{lepton}` jet and the lepton
#. :math:`\Delta R` between the two :math:`W` jets
#. :math:`\Delta R` between the first :math:`W` jet and the :math:`top_{hadron}` jet
#. :math:`\Delta R` between the second :math:`W` jet and the :math:`top_{hadron}` jet (should have same distribution as previous feature)
#. Combined mass of the :math:`top_{lepton}` jet and the lepton
#. Combined mass of the two :math:`W` jets
#. Combined mass of the two :math:`W` jets and the :math:`top_{hadron}` jet (reconstructed top mass)
#. Combined :math:`p_T` of the two :math:`W` jets and the :math:`top_{hadron}` jet
#. :math:`p_T` of the first :math:`W` jet
#. :math:`p_T` of the second :math:`W` jet (should have same distribution as previous feature)
#. :math:`p_T` of the :math:`top_{hadron}` jet
#. :math:`p_T` of the :math:`top_{lepton}` jet
#. ``btagCSVV2`` of the first :math:`W` jet (:math:`b`-tag value)
#. ``btagCSVV2`` of the second :math:`W` jet (should have same distribution as previous feature)
#. ``btagCSVV2`` of the :math:`top_{hadron}` jet
#. ``btagCSVV2`` of the :math:`top_{lepton}` jet
#. ``qgl`` of the first :math:`W` jet (quark-gluon discriminator)
#. ``qgl`` of the second :math:`W` jet (should have same distribution as previous feature)
#. ``qgl`` of the :math:`top_{hadron}` jet
#. ``qgl`` of the :math:`top_{lepton}` jet

For each permutation, all 20 features are fed into a boosted decision tree, which was trained to select correct permutations. After this, the permutation with the highest BDT score is selected as "correct", then we use those jet-parton assignments to calculate the observables of interest.

It is a future goal to move onto a more sophisticated architecture, as the BDT method is restrictive since it becomes computationally expensive for events with high jet multiplicity.

BDT Performance
---------------------------------------------------------------
We can first qualitatively compare the top mass reconstruction by the trijet combination method and the BDT method by comparing their distributions to the truth top mass reconstruction distribution. The events considered here are those in which it is possible to correctly reconstruct all jet-parton assignments in the leading four jets.

.. image:: images/topmassreconstruction.png
  :width: 80%
  :alt: Distribution of reconstructed top mass from different methods.
  
We can see that the result from using the BDT method (green) more closely matches the truth distribution (blue) than the trijet combination method (orange).

If we look into the performance by calculating which jet-parton assignments are predicted correctly, we also see that the BDT method performs better. If we look at the top 6 jets in each event and restrict the set of events to those in which full reconstruction is possible (i.e. all truth labels are present in the top 6 jets), we see that the BDT selects the correct three jets for the top mass reconstruction 60.10% of the time, while the trijet combination method only selects the correct three jets 28.31% of the time.

.. image:: images/bdt_performance_comparison.png
  :width: 80%
  :alt: Comparison of the BDT method to the trijet combination method.
  
The BDT is also trying to predict more information than the trijet combination method. Instead of finding the three correct jets to use for the reconstructed mass, we want to choose correct labels for four jets in an event. So to ensure that the BDT is performing as it should, we can compare BDT output to random chance. If we again look at the top 6 jets in each event and restrict the set of events to those in which full reconstruction is possible, we see the following:

.. image:: images/bdt_performance.png
  :width: 80%
  :alt: Comparison of the BDT output to random chance.

The BDT does much better than random chance at predicting jet-parton assignments.