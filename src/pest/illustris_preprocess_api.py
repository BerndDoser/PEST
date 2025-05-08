import os

import h5py
import numpy as np
import requests
from tqdm import tqdm


def get_illustris_api_key():
    """Return the API key for the IllustrisTNG API.
    The key can be set as an environment variable or in a file."""
    if "ILLUSTRIS_API_KEY" in os.environ:
        return os.environ["ILLUSTRIS_API_KEY"]
    elif os.path.isfile(".illustris_api_key.txt"):
        with open(".illustris_api_key.txt", "r") as file:
            return file.read().rstrip()
    raise ValueError(
        "No API key found. Please set the ILLUSTRIS_API_KEY environment variable"
        "or create a file named '.illustris_api_key.txt' with the API key"
    )


def get(path, key, params=None):
    """Get data from the IllustrisTNG API.
    Args:
        path (str): API endpoint.
        key (str): API key.
        params (dict, optional): Query parameters. Defaults to None.
    Returns:
        dict: Parsed JSON response.
    """

    print(f"GET {path}")

    headers = {"api-key": key}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers["content-type"] == "application/json":
        return r.json()  # parse json responses automatically

    if "content-disposition" in r.headers:
        filename = r.headers["content-disposition"].split("filename=")[1]
        with open(filename, "wb") as f:
            f.write(r.content)
        return filename  # return the filename string

    return r


def data_preprocess_api(
    sim="TNG100-1",
    snapshot=99,
    objects="centrals",
    selection_type="stellar mass",
    min_mass=1e8,
    max_mass=np.inf,
    component="stars",
    output_type="optical image",
    operations=["sum"],
    fov=1.5,
    fov_unit="kpc",
    depth=4,  # [particles]
    size=128,
    smoothing="None",  # [kpc]
    channels=["Masses"],
    bands=["u", "g", "r"],
    scale="log",
    stretch=0.02,
    Q=8.0,
    orientation="original",
    spin_aperture=30.0,  # [kpc]
    catalog_fields=["SubhaloStarMetallicity", "SubhaloSFR"],
    resolution_limit=1e9,  # [Msun]
    output_path="./images/",
    output_format="png",
    debug=False,
):
    """Preprocess data using IllustrisTNG API.

    Args:
        sim (str): Name of the simulation. Default is "TNG100-1".
        snapshot (int): Snapshot number. Default is 99.
        objects (str): Type of objects to process. Default is "centrals".
        selection_type (str): Type of selection. Default is "stellar mass".
        min_mass (float): Minimum mass for selection. Default is 1e8.
        max_mass (float): Maximum mass for selection. Default is np.inf.
        component (str): Component to process. Default is "stars".
        output_type (str): Type of output. Default is "optical image".
        operation (str): Operation to perform. Default is "sum".
        fov (float): Field of view in kpc. Default is 'scaled'.
        fov_unit (str): Unit of field of view. Default is 'kpc'.
        depth (int): Image depth in particles. Default is 4.
        size (int): Image size. Default is 128.
        smoothing (float): Smoothing radius in pixels. Default is 'None'.
        channels (int): List of image channels. Default is ["Masses"].
        scale (str): Image scaling. Default is "log".
        orientation (str): Orientation of the image. Default is "face-on".
        spin_aperture (float): Aperture for spin calculation in kpc. Default is 30.0.
        catalog_fields (list): List of fields to output in catalog. Default is ["SubhaloStarMetallicity", "SubhaloSFR"].
        resolution_limit (float): Resolution limit in Msun. Default is 1e9.
        output_path (str): Output path for images. Default is "./images/".
        debug (bool): Debug mode flag. Default is False.

    Returns:
        dict: Processed catalog data.
    """

    print(
        f"Parameters:\n"
        f" simulation: {sim}\n"
        f" snapshot: {snapshot}\n"
        f" objects: {objects}\n"
        f" selection_type: {selection_type}\n"
        f" min_mass: {min_mass:.2e} Ms, max_mass: {max_mass:.2e} Ms\n"
        f" component: {component}\n"
        f" output_type: {output_type}\n"
        f" operations: {operations}\n"
        f" fov: {fov}\n"
        f" fov_unit: {fov_unit}\n"
        f" depth: {depth} particles\n"
        f" size: {size} pixels\n"
        f" smoothing: {smoothing} pixels\n"
        f" channels: {channels}\n"
        f" bands: {bands}\n"
        f" scale: {scale}\n"
        f" Q: {Q}\n"
        f" range: {range}\n"
        f" orientation: {orientation}\n"
        f" spin_aperture: {spin_aperture:.1f} kpc\n"
        f" catalog_fields: {catalog_fields}\n"
        f" resolution_limit: {resolution_limit:.2e} Ms\n"
        f" output_path: {output_path}\n"
    )

    global mass_units_msun, dist_units_kpc

    assert output_type in [
        "optical image",
        "2D map",
        "PPP cube",
        "PPV cube",
        "none",
    ], f"{output_type} is not a valid output type."

    if objects == "centrals":
        primary_flag = [1]
    if objects == "satellites":
        primary_flag = [0]
    if objects == "all":
        primary_flag = [0, 1]

    if component == "stars":
        ptype = 4
    if component == "gas":
        ptype = 0
    if component == "dm":
        ptype = 1

    # if field == "Masses":
    #     comp_list = "Coordinates,Velocities,Masses"
    # else:
    #     comp_list = "'Coordinates,Velocities,Masses," + field + "'"
    # print("particle fields:", comp_list)

    # define API url
    url = "http://www.tng-project.org/api/" + sim + "/"

    # get api key
    api_key = get_illustris_api_key()

    # get header info
    sim_info = get(url, api_key)
    print(sim_info)
    box_size = sim_info["boxsize"] / sim_info["hubble"]
    print(f"Box size:", box_size / 1e3, " Mpc")
    mass_units_msun = 1e10 / sim_info["hubble"]
    dist_units_kpc = 1.0 / sim_info["hubble"]

    if selection_type == "total mass":
        sorting = "'-mass'"
    if selection_type == "stellar mass":
        sorting = "'-mass_stars'"
    print(f"\nSorting halos by {sorting}")
    subhalos = get(
        url + "snapshots/" + str(snapshot) + "/subhalos/",
        api_key,
        {"limit": 10000, "order_by": sorting},
    )
    print(f"Number of subhalos in catalog: {subhalos['count']}\n")

    print(f"selecting only {objects} with {selection_type} > {resolution_limit:.2e} Ms")
    print(f" and within {selection_type} range {min_mass:.2e} < M/Ms < {max_mass:.2e}")

    # form the search_query string
    search_query = (
        "?mass_stars__gt="
        + str(min_mass / mass_units_msun)
        + "&mass_stars__lt="
        + str(max_mass / mass_units_msun)
    )
    subhalos = get(
        url + "snapshots/" + str(snapshot) + "/subhalos/" + search_query,
        api_key,
        {"order_by": sorting},
    )
    print(f"\nNumber of subhalos in mass range: {subhalos['count']}\n")
    print(type(subhalos["count"]))

    # Loop over subhalos to read galaxy properties and particle data
    subid = []
    groupid = []
    m_stars = []
    m_halo = []
    m_tot = []
    r_half = []
    var0 = []
    var1 = []

    for i in range(subhalos["count"]):

        if debug:
            print(subhalos["results"][i]["url"])

        subhalo = get(subhalos["results"][i]["url"], api_key)
        mass_stars = subhalo["mass_stars"] * mass_units_msun
        mass_tot = subhalo["mass"] * mass_units_msun

        # sub_details = get(subhalo['meta']['url']+'info.json')
        sub_details = get(subhalo["meta"]["info"], api_key)
        v0 = sub_details[catalog_fields[0]]  # raw simulation units
        v1 = sub_details[catalog_fields[1]]

        group = get(subhalo["related"]["parent_halo"] + "info.json", api_key)
        mass_halo = group["Group_M_Crit200"] * mass_units_msun
        rvir_halo = group["Group_R_Crit200"] * dist_units_kpc

        if selection_type == "stellar mass":
            mass = mass_stars
        if selection_type == "total mass":
            mass = mass_tot

        if subhalo["subhaloflag"] != 1:
            continue
        if mass > max_mass:
            continue
        if mass < min_mass or mass < resolution_limit:
            break
        if subhalo["primary_flag"] not in primary_flag:
            continue

        subid.append(subhalo["id"])
        groupid.append(subhalo["grnr"])
        m_stars.append(mass_stars)
        m_tot.append(mass_tot)
        m_halo.append(mass_halo)
        r_half.append(subhalo["halfmassrad_stars"] * dist_units_kpc)
        var0.append(v0)
        var1.append(v1)

        # print galaxy info
        print("\n Galaxy:", i, " groupID:", groupid[-1], " subID:", subid[-1])
        print(
            f" Mstars = {m_stars[-1]:.2e} Ms, Rhalf = {r_half[-1]:.2f} kpc, Mtot = {mass_tot:.2e} Ms"
        )

        # load galaxy particles
        # cutout = get(subhalo["cutouts"]["subhalo"], api_key, {component: comp_list})  # to load only specific components
        cutout = get(subhalo["cutouts"]["subhalo"], api_key)

        if debug:
            print(f" Npart:{subhalo['len_stars']}")

        subhalo_pos = (
            np.array([subhalo["pos_x"], subhalo["pos_y"], subhalo["pos_z"]])
            * dist_units_kpc
        )
        subhalo_vel = np.array([subhalo["vel_x"], subhalo["vel_y"], subhalo["vel_z"]])
        if debug:
            print(
                f" subhalo position: {subhalo_pos[0]:.2f},{subhalo_pos[1]:.2f},{subhalo_pos[2]:.2f}"
            )

        particles = {}
        with h5py.File(cutout, "r") as f:
            # Iterate through the items in the file and convert to dictionary
            for key in f["PartType" + str(ptype)].keys():
                data = f["PartType" + str(ptype)][key][()]
                particles[key] = data

        # delete cutout file
        os.remove(cutout)

        # center coordinates and adjust for periodic boundaries
        adjusted_coordinates = particles["Coordinates"] * dist_units_kpc - subhalo_pos
        adjusted_coordinates = (
            np.mod(adjusted_coordinates + box_size / 2.0, box_size) - box_size / 2.0
        )
        particles["Coordinates"] = adjusted_coordinates
        # center velocities
        particles["Velocities"] = particles["Velocities"] - subhalo_vel
        particles["Masses"] = particles["Masses"] * mass_units_msun

        print(f" number of {component} particles: { len(particles['Masses']) }")
        print(f" total particle mass: {np.sum(particles['Masses']):.1e} Ms")

        # # rotate galaxy to desired orientation based on disk spin
        # particles = rotate_galaxy(particles, orientation, spin_aperture)

        # # create and save 2D map
        # if output_type == "2D map":
        #     print(" creating 2D map...")
        #     create_2Dmap(
        #         particles,
        #         operations,
        #         fov,
        #         fov_unit,
        #         depth,
        #         scale,
        #         size,
        #         smoothing,
        #         channels,
        #         subid[-1],
        #         component,
        #         orientation,
        #         output_path,
        #         output_format,
        #         debug,
        #     )

        # # create and save optical image
        # if output_type == "optical image":
        #     print(" creating image...")
        #     create_opticalimage(
        #         particles,
        #         fov,
        #         fov_unit,
        #         size,
        #         smoothing,
        #         bands,
        #         scale,
        #         stretch,
        #         Q,
        #         subid[-1],
        #         orientation,
        #         output_path,
        #         output_format,
        #         debug,
        #     )

        # # create and save PPP cube
        # if output_type == "PPP cube":
        #     print(" creating PPP cube...")
        #     create_cube_PPP(
        #         particles,
        #         operations,
        #         fov,
        #         depth,
        #         scale,
        #         size,
        #         smoothing,
        #         channels,
        #         subid[-1],
        #         component,
        #         orientation,
        #         output_path,
        #         debug,
        #     )

        # if output_type == "PPV cube":
        #     print(" creating PPV cube...")
        #     create_cube_PPV(
        #         particles,
        #         operations,
        #         fov,
        #         depth,
        #         scale,
        #         size,
        #         smoothing,
        #         channels,
        #         subid[-1],
        #         component,
        #         orientation,
        #         output_path,
        #         debug,
        #     )

        # if output_type == "point cloud":
        #     print(" creating point cloud...")
        #     pointcloud = [
        #         particles["Coordinates"],
        #         particles["Velocities"],
        #         particles[field],
        #     ]

    print("\nCreating catalog...")
    catalog_props = [
        "SubID",
        "GroupID",
        "logMstar",
        "logMtot",
        "logMhalo",
        "Rhalf",
    ] + catalog_fields
    print(" properties:", catalog_props)
    array_list = [
        subid,
        groupid,
        np.log10(m_stars),
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

    return catalog
