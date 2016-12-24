"""
Core phyutil model
==================

Defines the fundamental accelerator simulation related routines
Imported from NSLS II APHLA

:author: Lingyun Yang

:modified: Guobao Shen

"""

import logging

from .impact import build_result as build_impact_result
from phantasy import facility as machine 

_logger = logging.getLogger(__name__)


class Model:
    """ Model results
    """
    def __init__(self, simulation="IMPACT", resultdir=None):
        """class constructor.
        This is an online model interface, which assumes all elements have a unique name.
        
        :param simulation: simulation code name, either "IMPACT", "TLM", or others
        :param resultdir:  directory where simulation results are stored
        """
        self.code = simulation.upper()
        self.modelresult=None
        self.resultdir = resultdir
    
    def __getModelSeq4Elems(self, elems):
        """Get model sequence number
        
        :param elems: list of element object
        """
        def lastindex(ordermap):
            lastidx = -1
            for indexes in ordermap.values():
                for idx in indexes:
                    if idx > lastidx:
                        lastidx = idx
            return lastidx
        if isinstance(elems, (list, tuple)):
            res = []
            for el in elems:
                res.append(lastindex(machine._lat.latticemodelmap[el.name]))
            return res            
        else:
            return lastindex(machine._lat.latticemodelmap[elems.name])
    
    def _buildModelResult(self):
        """
        """
        if self.code == "IMPACT":
            self.modelresult = build_impact_result(directory=self.resultdir)
        else:
            raise Exception("Simulation code {0} not supported yet".format(self.code))
    
    def updateModelResult(self):
        """Update model result
        
        :return: Model object
        """
        return self.modelresult.updateResult()
    
    def getEnergy(self, elems=None):
        """get current sub-machine beam energy.
        Energy for a storage ring, or energy at given device for accelerating machine.
        If device is None, then final deliver energy.
        MeV for electron machine, or MeV/u for a heavy ion machine.
        
        :param elems: list of element name(s)
        :return: beam energy at the end of each element
    
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getEnergy(elemIdx)
    
    def getSPosition(self, elems=None):
        """Get element s position at the end if elemIdx is given, or list of s position for all totalelements
        
        :param elems: list of element name(s)
        
        :return: s position or list
        :raise: RuntimeError
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getSPosition(elemIdx)

    def getAbsPhase(self, elems=None):
        """Get accumulated beam phase in radian at the end if elemIdx is given, 
        or a list for all totalelements
        
        :param elems: list of element name(s)
        
        :return: accumulated beam phase or list
        :raise: RuntimeError
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getAbsPhase(elemIdx)
    
    def getBeta(self, elems=None):
        """Get beam beta (v/c) at the end if elemIdx is given, or a list for all totalelements
        
        :param elems: list of element name(s)
        
        :return: beta or list
        :raise: RuntimeError
        """
        if self.modelresult is None:
            self._buildModelResult()
        
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getBeta(elemIdx)        

    def getGamma(self, elems=None):
        """Get beam gamma at the end if elemIdx is given, or a list for all totalelements
        
        :param elems: list of element name(s)
        
        :return: gamma or list
        :raise: RuntimeError
        """
        if self.modelresult is None:
            self._buildModelResult()
        
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getGamma(elemIdx)        

    def getOrbit(self, plane="X", elems=None):
        """Get beam position at the end of an element if elemIdx is given, or beam orbit at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z" or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam position at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getOrbit(plane, elemIdx)

    def getTwissAlpha(self, plane="X", elems=None):
        """Get beam twiss alpha parameters at the end of an element if elemIdx is given, or at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z", or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam twiss alpha at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getTwissAlpha(plane, elemIdx)

    def getTwissBeta(self, plane="X", elems=None):
        """Get beam twiss beta parameters at the end of an element if elemIdx is given, or at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z", or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam twiss beta at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getTwissBeta(plane, elemIdx)
        
    def getBeamRms(self, plane="X", elems=None):
        """Get beam RMS parameters at the end of an element if elemIdx is given, or at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z", or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam twiss beta at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getBeamRms(plane, elemIdx)

    def getEmittance(self, plane="X", elems=None):
        """Get beam normalized emittance (m-rad for transverse and degree-MeV for longitudinal)
        at the end of an element if elemIdx is given, or at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z", or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam emittance at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getEmittance(plane, elemIdx)
    
    def getBeamMomentumCentroid(self, plane="X", elems=None):
        """Get beam centroid momentum (radian for transverse and MeV for longitudinal)
        at the end of an element if elemIdx is given, or at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z", or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam emittance at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getBeamMomentumCentroid(plane, elemIdx)
        
    def getMomentumRms(self, plane="X", elems=None):
        """Get beam RMS momentum (radian for transverse and MeV for longitudinal)
        at the end of an element if elemIdx is given, or at all totalelements.
        Current implementation returns all position information from simulation, and does not separate BPM 
        from other devices like magnet and other diagnostic devices like profile monitor.

        :param plane:    beam plane, either "X", "Y", "Z", or "XY"
        :param elems: list of element name(s), `None` by default 
        
        :return: beam emittance at given location, or at all totalelements
        """
        if self.modelresult is None:
            self._buildModelResult()
        if elems is None:
            elemIdx = None
        else:
            elemIdx = self.__getModelSeq4Elems(elems)
        return self.modelresult.getMomentumRms(plane, elemIdx)    


    def saveModel(self, name, latname, latbranch, latversion, description="",
                    username=None, password=None):
        """Save the model to the Lattice/Model store or client configured for
        the current submachine. The lattice corresponding to ths this model
        must be specifed by name, branch and version. Optionally specify a
        username and password for authenticating the store or client.

        :param name: model name
        :param latname: lattice name
        :param latbranch: lattice branch
        :param latversion: lattice version
        :param description: optional model description
        :param username: optional username for service authentication
        :param password: optional password for service authentication
        """
        store = machine.getLatticeModelStore(username=username, password=password)

        model = {
            'description':description,
            'simulationCode':'impact',
            'simulationAlgorithm':'lorentz'
        }

        energy = self.getEnergy()
        position = self.getSPosition()
        x = self.getOrbit("X")
        y = self.getOrbit("Y")
        z = self.getOrbit("Z")
        xrms = self.getBeamRms("X")
        yrms = self.getBeamRms("Y")
        zrms = self.getBeamRms("Z")
        xp = self.getBeamMomentumCentroid("X")
        yp = self.getBeamMomentumCentroid("Y")
        zp = self.getBeamMomentumCentroid("Z")
        xprms = self.getMomentumRms("X")
        yprms = self.getMomentumRms("Y")
        zprms = self.getMomentumRms("Z")
        beamparams = {}
        for elem, elemorder in machine._lat.latticemodelmap.iteritems():
            for order, indexes in elemorder.iteritems():
                for idx in indexes:
                    beamparams[order] = {
                            'name':elem,
                            'energy':energy[idx],
                            'position':position[idx],
                            'x':x[idx], 'y':y[idx], 'z':z[idx],
                            'xrms':xrms[idx], 'yrms':yrms[idx], 'zrms':zrms[idx],
                            'xp':xp[idx], 'yp':yp[idx], 'zp':zp[idx],
                            'xprms':xprms[idx], 'yprms':yprms[idx], 'zprms':zprms[idx],
                            'transferMatrix':[]
                        }

        model['beamParameter'] = beamparams

        store.saveModel(latname, latbranch, latversion, { name:model })

