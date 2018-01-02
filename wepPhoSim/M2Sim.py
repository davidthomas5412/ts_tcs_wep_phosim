import os
import numpy as np

from wepPhoSim.MirrorSim import MirrorSim
from wepPhoSim.CoTransform import M2CRS2ZCRS

class M2Sim(MirrorSim):

    def getActForce(self, actForceFileName="M2_1um_force.DAT"):
        """
        
        Get the mirror actuator forces in N.
        
        Keyword Arguments:
            actForceFileName {str} -- Actuator force file name. (default: {"M2_1um_force.DAT"})
        
        Returns:
            [ndarray] -- Actuator forces in N.
        """
        
        forceInN = self.getMirrorData(actForceFileName)

        return forceInN

    def getPrintthz(self, zAngleInRadian, preCompElevInRadian=0, FEAfileName="M2_GT_FEA.txt"):
        """
        
        Get the mirror print in um along z direction in specific zenith angle. 
        
        Arguments:
            zAngleInRadian {[float]} -- Zenith angle in radian.
        
        Keyword Arguments:
            preCompElevInRadian {float} -- Pre-compensation elevation angle in radian. (default: {0})
            FEAfileName {str} -- Finite element analysis (FEA) model data file name. (default: {"M2_GT_FEA.txt"})
        
        Returns:
            [ndarray] -- Corrected projection in um along z direction.
        """

        # Read the FEA file
        data = self.getMirrorData(FEAfileName, skiprows=1)

        # Zenith direction in um
        zdz = data[:, 2]

        # Horizon direction in um
        hdz = data[:, 3]

        # Do the M2 gravitational correction.
        # Map the changes of dz on a plane for certain zenith angle 
        printthzInUm = zdz * np.cos(zAngleInRadian) + hdz * np.sin(zAngleInRadian)
        
        # Do the pre-compensation elevation angle correction
        printthzInUm -= zdz * np.cos(preCompElevInRadian) + hdz * np.sin(preCompElevInRadian)
        
        return printthzInUm

    def getTempCorr(self, M2TzGrad, M2TrGrad, FEAfileName="M2_GT_FEA.txt"):
        """
        
        Get the mirror print correction in um along z direction for certain temperature gradient. 
        
        Arguments:
            M2TzGrad {[float]} -- Temperature gradient along z direction. (+/-2sigma spans 1C).
            M2TrGrad {[float]} -- Temperature gradient along r direction. (+/-2sigma spans 1C).
        
        Keyword Arguments:
            FEAfileName {str} -- Finite element analysis (FEA) model data file name. 
                                 (default: {"M2_GT_FEA.txt"})
        
        Returns:
            [ndarray] -- Corrected projection in um along z direction.
        """
        
        # Read the FEA file
        data = self.getMirrorData(FEAfileName, skiprows=1)

        # Z-gradient in um
        tzdz = data[:, 4]

        # r-gradient in um
        trdz = data[:, 5]

        # Get the temprature correction
        tempCorrInUm = M2TzGrad * tzdz + M2TrGrad * trdz

        return tempCorrInUm

    def getMirrorResInZemax(self, gridFileName="M2_1um_grid.DAT", numTerms=22, writeZcToFilePath=None):
        """
        
        Get the residue of surface (mirror print along z-axis) in um after the fitting with spherical 
        Zernike polynomials (zk) under the Zemax coordinate.
        
        Keyword Arguments:
            gridFileName {str} -- File name of bending mode data. (default: {"M2_1um_grid.DAT"})
            numTerms {int} -- Number of Zernike terms to fit. (default: {22})
            writeZcToFilePath {[str]} -- File path to write the fitted zk. (default: {None})
        
        Returns:
            [ndarray] -- Fitted residue in um after removing the fitted zk terms in Zemax coordinate.
            [ndarray] -- X position in m in Zemax coordinate. 
            [ndarray] -- Y position in m in Zemax coordinate.
            [ndarray] -- Fitted zk in um in Zemax coordinate.
        """

        # Get the bending mode information
        data = self.getMirrorData(gridFileName)

        # Get the x, y coordinate
        bx = data[:, 0]
        by = data[:, 1]

        # Transform the M2 coordinate to Zemax coordinate 
        bxInZemax, byInZemax, surfInZemax = M2CRS2ZCRS(bx, by, self.surf)

        # Get the mirror residue and zk in um
        resInUmInZemax, zcInUmInZemax = self._MirrorSim__getMirrorResInNormalizedCoor(surfInZemax, 
                bxInZemax/self.RinM, byInZemax/self.RinM, numTerms, writeZcToFilePath=writeZcToFilePath)

        return resInUmInZemax, bxInZemax, byInZemax, zcInUmInZemax

    def writeMirZkAndGridResInZemax(self, resFile=None, surfaceGridN=200, gridFileName="M2_1um_grid.DAT", 
                                    numTerms=22, writeZcToFilePath=None):
        """
        
        Write the grid residue in mm of mirror surface after the fitting with Zk under the Zemax 
        coordinate.
        
        Keyword Arguments:
            surfaceGridN {int} -- Surface grid number. (default: {200})
            gridFileName {str} -- File name of bending mode data. (default: {"M2_1um_grid.DAT"})
            numTerms {int} -- Number of Zernike terms to fit. (default: {22})
            writeZcToFilePath {[str]} -- [File path to write the fitted zk. (default: {None})
            resFile {[str]} -- File path to save the grid surface residue map. (default: {None})

        Returns:
            [str] -- Grid residue map related data.
        """

        # Get the residure map
        resInUmInZemax, bxInMinZemax, byInMinZemax = self.getMirrorResInZemax(gridFileName=gridFileName, 
                                                        numTerms=numTerms, writeZcToFilePath=writeZcToFilePath)[0:3]

        # Change the unit
        zfInMm = resInUmInZemax * 1e-3
        xfInMm = bxInMinZemax * 1e3
        yfInMm = byInMinZemax * 1e3
        innerRinMm = self.RiInM * 1e3
        outerRinMm = self.RinM * 1e3

        # Get the residue map used in Zemax
        # Content header: (NUM_X_PIXELS, NUM_Y_PIXELS, delta x, delta y)
        # Content: (z, dx, dy, dxdy)
        content = self._MirrorSim__gridSampInMnInZemax(zfInMm, xfInMm, yfInMm, innerRinMm, outerRinMm, 
                                                        surfaceGridN, surfaceGridN, resFile=resFile)

        return content

    def showMirResMap(self):

        pass

if __name__ == "__main__":
    
    # Inner raidus
    innerRinM = 0.9

    # Outer raidus
    outerRinM = 1.71

    # M2 data directory
    mirrorDataDir = "../data/M2"

    # Instantiate mirror
    M2 = M2Sim(innerRinM, outerRinM)
    M2.setMirrorDataDir(mirrorDataDir)

    forceInN = M2.getActForce()

    zAngleInDeg = 27.0912
    zAngleInRadian = zAngleInDeg/180*np.pi
    printthzInUm = M2.getPrintthz(zAngleInRadian)

    M2TzGrad = -0.0675
    M2TrGrad = -0.1416
    tempCorrInUm = M2.getTempCorr(M2TzGrad, M2TrGrad)

    mirrorSurfInUm = printthzInUm + tempCorrInUm
    M2.setSurfAlongZ(mirrorSurfInUm)

    # Get the residue and zc
    writeZcToFilePath = "/Users/Wolf/Desktop/tempZc.txt"

    # Need to update zernike fit to 28 terms
    # resInUmInZemax, bxInZemax, byInZemax, zcInUmInZemax = M2.getMirrorResInZemax(numTerms=22, writeZcToFilePath=writeZcToFilePath)

    # Write the grid surface residue map
    resFile = "/Users/Wolf/Desktop/resM2.txt"
    # content = M2.writeMirZkAndGridResInZemax(resFile=resFile, numTerms=28, writeZcToFilePath=writeZcToFilePath)

    # Show the grid mirror residue map
    


