import illustris_python as il
import numpy as np


def data_preprocess_local(
    sim="TNG100-1",
    snapshot=99,
    filepath="./sims.TNG/",
    objects="centrals",
    selection_type="stellar mass",
    min_mass=1e8,
    max_mass=np.inf,
    component="stars",
    output_type="2D projection",
    field="Masses",
    operation="sum",
    fov=None,  # [kpc]
    image_depth=1,  # [particles]
    image_size=128,
    smoothing="None",  # [kpc]
    channels=1,
    image_scale="log",
    orientation="face-on",
    spin_aperture=30.0,  # [kpc]
    catalog_fields=["SubhaloStarMetallicity", "SubhaloSFR"],
    resolution_limit=1e9,  # [Msun]
    output_path="./images/",
    debug=False,
):
    """Preprocess local simulation data.

    Args:
        sim (str): Name of the simulation.
        snapshot (int): Snapshot number.
        filepath (str): Path to the simulation files.
        objects (str): Type of objects to consider (centrals, satellites, all).
        selection_type (str): Type of selection (stellar mass, total mass).
        min_mass (float): Minimum mass for selection.
        max_mass (float): Maximum mass for selection.
        component (str): Component to analyze (stars, gas, dm).
        output_type (str): Type of output (2D projection, point cloud).
        field (str): Field to analyze.
        operation (str): Operation to perform.
        fov (float): Field of view in kpc.
        image_depth (int): Image depth in particles.
        image_size (int): Image size.
        smoothing (float): Smoothing factor in kpc.
        channels (int): Number of channels.
        image_scale (str): Image scaling method.
        orientation (str): Orientation of the image.
        spin_aperture (float): Spin aperture in kpc.
        catalog_fields (list): List of catalog fields.
        resolution_limit (float): Resolution limit in Msun.
        output_path (str): Path to save the output images.
        debug (bool): Enable debug mode.

    Returns:
        dict: Catalog containing selected properties for each object.
    """

    print(
        f"Parameters:\n"
        f" simulation: {sim}\n"
        f" snapshot: {snapshot}\n"
        f" filepath: {filepath}\n"
        f" objects: {objects}\n"
        f" selection_type: {selection_type}\n"
        f" min_mass: {min_mass:.2e} Ms, max_mass: {max_mass:.2e} Ms\n"
        f" component: {component}\n"
        f" output_type: {output_type}\n"
        f" field: {field}\n"
        f" operation: {operation}\n"
        f" fov: {fov}\n"
        f" image_depth: {image_depth} particles\n"
        f" image_size: {image_size} pixels\n"
        f" smoothing: {smoothing} kpc\n"
        f" channels: {channels}\n"
        f" image_scale: {image_scale}\n"
        f" orientation: {orientation}\n"
        f" spin_aperture: {spin_aperture:.1f} kpc\n"
        f" catalog_fields: {catalog_fields}\n"
        f" resolution_limit: {resolution_limit:.2e} Ms\n"
        f" output_path: {output_path}\n"
    )

    global mass_units_msun, dist_units_kpc

    if component == "stars":
        ptype = 4
    if component == "gas":
        ptype = 0
    if component == "dm":
        ptype = 1

    if objects == "centrals":
        primary_flag = [1]
    if objects == "satellites":
        primary_flag = [0]
    if objects == "all":
        primary_flag = [0, 1]

    # define path to data
    basePath = filepath + sim + "/output/"

    # define conversions to physical units
    header = il.groupcat.loadHeader(basePath, 99)
    mass_units_msun = 1e10 / header["HubbleParam"]
    dist_units_kpc = header["Time"] / header["HubbleParam"]

    # load subhalos (i.e. galaxies)
    print(f"loading galaxy catalog from {basePath}...")
    subhalos = il.groupcat.loadSubhalos(basePath, snapshot)
    subhalos["SubhaloID"] = np.arange(subhalos["count"])

    # Loop over galaxies to read galaxy properties and particle data
    sub_id = []
    group_id = []
    m_stellar = []
    m_halo = []
    m_tot = []
    r_half = []
    var0 = []
    var1 = []
    print(f"selecting only {objects} with {selection_type} > {resolution_limit:.2e} Ms")
    print(f" and within {selection_type} range {min_mass:.2e} < M/Ms < {max_mass:.2e}")

    # select mass range
    m_stars_all = subhalos["SubhaloMassType"][:, 4] * mass_units_msun
    m_tot_all = subhalos["SubhaloMass"] * mass_units_msun
    if selection_type == "stellar mass":
        mass_mask = (m_stars_all > min_mass) * (m_stars_all < max_mass)
    if selection_type == "total mass":
        mass_mask = (m_tot_all > min_mass) * (m_tot_all < max_mass)
    print(f" ... selected {sum(mass_mask)} subhalos in mass range")

    i = 0
    for subid in subhalos["SubhaloID"][mass_mask]:

        # subhalo properties
        subhalo = il.groupcat.loadSingle(basePath, snapshot, subhaloID=subid)
        ms = subhalo["SubhaloMassType"][4] * mass_units_msun
        mtot = subhalo["SubhaloMass"] * mass_units_msun
        rh = subhalo["SubhaloHalfmassRadType"][4] * dist_units_kpc
        v0 = subhalo[catalog_fields[0]]  # raw simulation units
        v1 = subhalo[catalog_fields[1]]

        # Group properties
        gid = subhalo["SubhaloGrNr"]
        group = il.groupcat.loadSingle(basePath, snapshot, haloID=gid)
        mh = group["Group_M_Crit200"] * mass_units_msun
        central_subid = group["GroupFirstSub"]
        if central_subid == subid:
            central_flag = 1
        else:
            central_flag = 0

        if selection_type == "stellar mass":
            mass = ms
        if selection_type == "total mass":
            mass = mtot

        if debug and subid % 1000 == 0:
            print(
                "\nGalaxy:",
                i,
                " subif:",
                subid,
                " gid:",
                gid,
                "flag:",
                central_flag,
                "mass:",
                np.log10(mass),
            )

        if subhalo["SubhaloFlag"] != 1:
            continue
        if mass < resolution_limit:
            continue
        if mass < min_mass:
            continue
        if mass > max_mass:
            continue
        if central_flag not in primary_flag:
            continue

        # print galaxy info
        print(
            "\nGalaxy:",
            i,
            " subID:",
            subid,
            " groupID:",
            gid,
            " primary_flag:",
            central_flag,
        )
        print(f" Mstars={ms:.2e} Ms, Rhalf={rh:.2f} kpc, Mhalo={mh:.2e} Ms")

        # load galaxy particles
        print(" loading particles...")
        particles = il.snapshot.loadSubhalo(
            basePath, snapshot, subid, component
        )  # all fields
        # print number of particles in the galaxy
        masses_temp = particles["Masses"] * mass_units_msun
        print(f" number of {component} particles: { len(masses_temp) }")
        print(f" total particle mass: {np.sum(masses_temp):.1e} Ms")

        # center the coordinates/velocities and masses
        particles["Coordinates"] = (
            particles["Coordinates"] - subhalo["SubhaloPos"]
        ) * dist_units_kpc
        particles["Velocities"] = particles["Velocities"] - subhalo["SubhaloVel"]
        particles["Masses"] = particles["Masses"] * mass_units_msun

        # rotate galaxy
        particles = rotate_galaxy(particles, orientation, spin_aperture)

        # create and save image
        if output_type == "2D projection":
            print(" creating image...")
            create_image(
                particles,
                field,
                operation,
                fov,
                image_depth,
                image_scale,
                image_size,
                smoothing,
                channels,
                subid,
                component,
                orientation,
                output_path,
                debug,
            )

        if output_type == "point cloud":
            print(" creating point cloud...")
            pointcloud = {
                "Coordinates": particles["Coordinates"],
                "Velocities": particles["Velocities"],
                field: particles[field],
            }

        sub_id.append(subid)
        group_id.append(gid)
        m_stellar.append(ms)
        m_halo.append(mh)
        m_tot.append(mtot)
        r_half.append(rh)
        var0.append(v0)
        var1.append(v1)
        i += 1

        del subhalo, group

    print("\nCreating catalog...")
    catalog_props = ["SubID", "GroupID", "logMstar", "logMtot", "logMhalo", "Rhalf"]
    catalog_props = catalog_props + catalog_fields
    print(" properties:", catalog_props)
    array_list = [
        sub_id,
        group_id,
        np.log10(m_stellar),
        np.log10(m_tot),
        np.log10(m_halo),
        r_half,
        var0,
        var1,
    ]
    catalog = {}
    for prop_name, array in zip(catalog_props, array_list):
        catalog[str(prop_name)] = array

    print("... done")

    del subhalos

    return catalog
