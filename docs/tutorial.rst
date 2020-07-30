Tutorial
========

.. |ModelFlame| replace:: :py:class:`ModelFlame <flame_utils.core.model.ModelFlame>`
.. |BeamState| replace:: :py:class:`BeamState <flame_utils.core.state.BeamState>`
.. |run()| replace:: :py:class:`run() <flame_utils.core.model.ModelFlame.run>`
.. |collect_data()| replace:: :py:class:`collect_data() <flame_utils.core.model.ModelFlame.collect_data>`
.. |hplot()| replace:: :py:func:`hplot() <flame_utils.viz.plotlat.hplot>`

.. |bmstate| replace:: :py:class:`bmstate <flame_utils.core.model.ModelFlame.bmstate>`
.. |set_twiss()| replace:: :py:class:`set_twiss() <flame_utils.core.state.BeamState.set_twiss>`
.. |find()| replace:: :py:class:`find() <flame_utils.core.model.ModelFlame.find>`
.. |get_element()| replace:: :py:class:`get_element() <flame_utils.core.model.ModelFlame.get_element>`
.. |reconfigure()| replace:: :py:class:`reconfigure() <flame_utils.core.model.ModelFlame.reconfigure>`
.. |generate_latfile()| replace:: :py:class:`generate_latfile() <flame_utils.core.model.ModelFlame.generate_latfile>`


1. Basic usage
--------------

In Python interface (include IPython-notebook),
user can import flame_utils `ModelFlame` class.
::

    from flame_utils import ModelFlame

Create |ModelFlame| object with input lattice file.
::

    fm = ModelFlame(lat_file = 'userfile.lat')

|run()| simulation through the lattice.
::

     r, s = fm.run()

Here ``s`` is the |BeamState| object at the end of the lattice section.

User can obtain various beam parameter which listed in |BeamState| docstring.
::

    s.xrms       # returns horizontal(x) rms beam size
    s.ref_IonEk  # returns reference beam energy

Plot result by using |hplot()|.
::

    from flame_utils import hplot
    hplot('xrms', 'yrms', machine=fm)

User can observe the beam state history by using monitor keyword in |run()|.
::

    r, s = fm.run(monitor = 'all')  # observe the beam state in all elements

Here ``r`` contains the list of (index, |BeamState|) for each lattice element.

Also, obtain list of beam parameters along the lattice by using |collect_data()|.
::

    d = fm.collect_data(r, 'pos', 'xrms', 'yrms')
    d['xrms']  # list of xrms along the lattice

Plot by using ``matplotlib``
::

    import matplotlib.pyplot as plt
    plt.plot(d['pos'], d['xrms'])
    plt.plot(d['pos'], d['xrms'])

or plot by using |hplot()|.
::

    from flame_utils import hplot
    hplot('xrms', 'yrms', machine=fm, result=r)


2. Initial beam parameter control
---------------------------------

|bmstate| returns initial |BeamState| of the input lattice file.
::

    fm.bmstate.xrms # initial xrms size

User can change the initial beam condition by using |set_twiss()|.
::

    fm.bmstate.set_twiss('x', rmssize = 0.7)
    # set xrms size to 0.7 mm with the same emittance


3. Lattice element parameter control
------------------------------------

|find()| index of the lattice element.
::

    fm.find(type='quadrupole')  # find by type
    >>> [4, 6, 8, 28, 30, 32, 44, 46, 48, 56, 58, 60, 64, 66, 68, 78, 80, 82, 84]
    fm.find(name='qd1')         # find by name
    >>> [78]

Check element information by |get_element()|.
::

    fm.get_element(index = 78)    # get by index
    # or
    fm.get_element(name = 'qd1')  # get by name
    >>> [{'index': 78,
        'properties': {'aper': 0.025,
        'name': 'qd1',
        'type': 'quadrupole',
        'L': 0.4107,
        'B2': 10.0}}]

Change the element parameter by |reconfigure()|.
::

    fm.reconfigure(78, {'B2': 8.0})     # set new parameter by index
    # or
    fm.reconfigure('qd1', {'B2': 8.0})  # set new parameter by name

Use can find all elements properies in :ref:`element` .

Save new setting to the file by |generate_latfile()|.
::

    fm.generate_latfile(latfile='New.lat', original='userfile.lat')


4. Advanced use cases
---------------------

In |run()| method, user can define ``from_element``,
``to_element`` and ``bmstate`` as an initial condition.
::

    r1, s1 = fm.run(to_element=56)               # simulate to 56th element
    r2, s2 = fm.run(bmstate=s1, from_element=57) # simulate from 57th element

This result should be the same as
::

    r, s = fm.run()


These are benefits to minimize the simulation time by selecting the re-calculation section.


Notes
-----

In the case of user put ``monitor = 'all'`` in |run()|,
the index of ``r`` is consistent with the lattice index in |get_element()|.
::

    r1, s1 = fm.run(monitor='all')
    r2, s2 = fm.run(to_element=78)

Here ``r1[78][1]`` is the same as ``s2``.
