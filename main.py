#Import module Transmitter
from transmitter import Transmitter as Transmitter
#Import module Surface
from surface import Surface as Surface



transmitter = Transmitter(
        "Led1",
        position=[2.5, 2.5, 3],
        normal=[0, 0, -1],
        mlambert=1,
        wavelengths=[620, 530, 475],
        fwhm=[20, 45, 20],
        modulation='ieee16',
        luminous_flux=5000
    )
transmitter.plot_spd_normalized()

surface = Surface(
    name="surfacePlaster",
    position=[1, 1, 0],
    normal=[0, 0, 1],
    reflectance='plaster',
    size=[0.2, 0.2]
    )
surface.plot_reflectance()

