import os, unittest
import numpy as np

from wepPhoSim.MirrorSim import MirrorSim
from wepPhoSim.CoTransform import M2CRS2ZCRS

class M2Sim(MirrorSim):

    def __init__(self, surf=None, mirrorDataDir=None):
        """
        
        Initiate the M2Sim object.
        
        Keyword Arguments:
            surf {[ndarray]} -- Mirror surface along z direction. (default: {None})
            mirrorDataDir {[str]} -- Mirror data directory. (default: {None})
        """

        # Outer radius of M2 mirror in m
        R = 1.710

        # Inner radius of M2 mirror in m
        Ri = 0.9

        super(M2Sim, self).__init__(Ri, R, surf=surf, mirrorDataDir=mirrorDataDir)

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
            M2TzGrad {[float]} -- Temperature gradient along z direction in degree C. (+/-2sigma spans 1C).
            M2TrGrad {[float]} -- Temperature gradient along r direction in degree C. (+/-2sigma spans 1C).

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

    def getMirrorResInMmInZemax(self, gridFileName="M2_1um_grid.DAT", numTerms=28, 
                                writeZcInMnToFilePath=None):
        """

        Get the residue of surface (mirror print along z-axis) in mm after the fitting with spherical
        Zernike polynomials (zk) under the Zemax coordinate.

        Keyword Arguments:
            gridFileName {str} -- File name of bending mode data. (default: {"M2_1um_grid.DAT"})
            numTerms {int} -- Number of Zernike terms to fit. (default: {28})
            writeZcInMnToFilePath {[str]} -- File path to write the fitted zk in mm. (default: {None})

        Returns:
            [ndarray] -- Fitted residue in mm after removing the fitted zk terms in Zemax coordinate.
            [ndarray] -- X position in mm in Zemax coordinate.
            [ndarray] -- Y position in mm in Zemax coordinate.
            [ndarray] -- Fitted zk in mm in Zemax coordinate.
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
                                                bxInZemax/self.RinM, byInZemax/self.RinM, numTerms)

        # Change the unit to mm
        resInMmInZemax = resInUmInZemax * 1e-3
        bxInMmInZemax = bxInZemax * 1e3
        byInMmInZemax = byInZemax * 1e3
        zcInMmInZemax = zcInUmInZemax * 1e-3

        # Save the file of fitted Zk
        if (writeZcInMnToFilePath is not None):
            np.savetxt(writeZcInMnToFilePath, zcInMmInZemax)

        return resInMmInZemax, bxInMmInZemax, byInMmInZemax, zcInMmInZemax

    def writeMirZkAndGridResInZemax(self, resFile=None, surfaceGridN=200, gridFileName="M2_1um_grid.DAT",
                                    numTerms=28, writeZcInMnToFilePath=None):
        """

        Write the grid residue in mm of mirror surface after the fitting with Zk under the Zemax
        coordinate.

        Keyword Arguments:
            resFile {[str]} -- File path to save the grid surface residue map. (default: {None})
            surfaceGridN {int} -- Surface grid number. (default: {200})
            gridFileName {str} -- File name of bending mode data. (default: {"M2_1um_grid.DAT"})
            numTerms {int} -- Number of Zernike terms to fit. (default: {28})
            writeZcInMnToFilePath {[str]} -- File path to write the fitted zk in mm. (default: {None})

        Returns:
            [str] -- Grid residue map related data.
        """

        # Get the residure map
        resInMmInZemax, bxInMmInZemax, byInMmInZemax = self.getMirrorResInMmInZemax(gridFileName=gridFileName,
                                                 numTerms=numTerms, writeZcInMnToFilePath=writeZcInMnToFilePath)[0:3]

        # Change the unit from m to mm
        innerRinMm = self.RiInM * 1e3
        outerRinMm = self.RinM * 1e3

        # Get the residue map used in Zemax
        # Content header: (NUM_X_PIXELS, NUM_Y_PIXELS, delta x, delta y)
        # Content: (z, dx, dy, dxdy)
        content = self._MirrorSim__gridSampInMnInZemax(resInMmInZemax, bxInMmInZemax, byInMmInZemax, innerRinMm,
                                                        outerRinMm, surfaceGridN, surfaceGridN, resFile=resFile)

        return content

    def showMirResMap(self, gridFileName="M2_1um_grid.DAT", numTerms=28, resFile=None, writeToResMapFilePath=None):
        """

        Show the mirror residue map.

        Keyword Arguments:
            gridFileName {str} -- File name of bending mode data. (default: {"M2_1um_grid.DAT"})
            numTerms {int} -- Number of Zernike terms to fit. (default: {28})
            resFile {[str]} -- File path of the grid surface residue map. (default: {None})
            writeToResMapFilePath {[str]} -- File path to save the residue map. (default: {None})
        """

        # Get the residure map
        resInMmInZemax, bxInMmInZemax, byInMmInZemax = self.getMirrorResInMmInZemax(gridFileName=gridFileName,
                                                                                     numTerms=numTerms)[0:3]

        # Change the unit
        outerRinMm = self.RinM * 1e3
        self._MirrorSim__showResMap(resInMmInZemax, bxInMmInZemax, byInMmInZemax, outerRinMm,
                                    resFile=resFile, writeToResMapFilePath=writeToResMapFilePath)

class M2SimTest(unittest.TestCase):
    
    """
    Test functions in M2Sim.
    """

    def setUp(self):

        # Directory of M2 data
        self.mirrorDataDir = os.path.join("..", "data", "M2")

    def testFunc(self):

        # Instantiate the M2Sim object
        M2 = M2Sim()

        self.assertEqual(M2.RiInM, 0.9)
        self.assertEqual(M2.RinM, 1.71)

        M2.setMirrorDataDir(self.mirrorDataDir)

        forceInN = M2.getActForce()
        self.assertEqual(forceInN.shape, (156, 156))

        zAngleInDeg = 27.0912
        zAngleInRadian = zAngleInDeg/180*np.pi
        printthzInUm = M2.getPrintthz(zAngleInRadian)

        ansFilePath = os.path.join("..", "testData", "testM2Func", "M2printthz.txt")
        ansPrintthzInUm = np.loadtxt(ansFilePath)
        self.assertLess(np.sum(np.abs(printthzInUm-ansPrintthzInUm)), 1e-10)

        M2TzGrad = -0.0675
        M2TrGrad = -0.1416
        tempCorrInUm = M2.getTempCorr(M2TzGrad, M2TrGrad)

        ansFilePath = os.path.join("..", "testData", "testM2Func", "M2tempCorr.txt")
        ansTempCorrInUm = np.loadtxt(ansFilePath)
        self.assertLess(np.sum(np.abs(tempCorrInUm-ansTempCorrInUm)), 1e-10)

        numTerms = 28
        mirrorSurfInUm = printthzInUm + tempCorrInUm
        M2.setSurfAlongZ(mirrorSurfInUm)
        zcInMmInZemax = M2.getMirrorResInMmInZemax(numTerms=numTerms)[3]

        ansFilePath = os.path.join("..", "testData", "testM2Func", "sim6_M2zlist.txt")
        ansZcInUmInZemax = np.loadtxt(ansFilePath)
        ansZcInMmInZemax = ansZcInUmInZemax*1e-3
        self.assertLess(np.sum(np.abs(zcInMmInZemax[0:numTerms]-ansZcInMmInZemax[0:numTerms])), 1e-9)

        resFile = os.path.join("..", "output", "M2res.txt")
        M2.writeMirZkAndGridResInZemax(resFile=resFile, numTerms=numTerms)
        content = np.loadtxt(resFile)

        ansFilePath = os.path.join("..", "testData", "testM2Func", "sim6_M2res.txt")
        ansContent = np.loadtxt(ansFilePath)
        self.assertLess(np.sum(np.abs(content[0,:]-ansContent[0,:])), 1e-9)
        self.assertLess(np.sum(np.abs(content[1:,0]-ansContent[1:,0])), 1e-9)

        writeToResMapFilePath = os.path.join("..", "output", "M2resMap.png")
        M2.showMirResMap(numTerms=numTerms, resFile=resFile, writeToResMapFilePath=writeToResMapFilePath)
        self.assertTrue(os.path.isfile(writeToResMapFilePath))

        os.remove(resFile)
        os.remove(writeToResMapFilePath)

if __name__ == "__main__":

    # Do the unit test
    unittest.main()
