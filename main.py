from surface import Surface as Surface

surface = Surface(
    name="surfacePlaster",
    position=[1, 1, 0],
    normal=[0, 0, 1],
    reflectance='plaster',
    size=[0.2, 0.2]
    )
surface.plot_reflectance()

