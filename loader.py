import numpy as np
import pandas as pd

from os.path import exists, expanduser, join, splitext
from warnings import warn

from air import air2vacuum
from basics import data_path
from hitran import fetch_hitran


def replace_PQR_with_m101(df):
    """Return P, Q, R in column ``branch`` with -1, 0, 1 to get a fully numeric
    database. This improves performances quite a lot, as Pandas doesnt have a
    fixed-string dtype hence would use the slow ``object`` dtype.
    Parameters
    ----------
    df: pandas Dataframe
        ``branch`` must be a column name.
    Returns
    -------
    None:
        ``df`` is is modified in place
    """

    # Note: somehow pandas updates dtype automatically. We have to check
    # We just have to replace the column:
    if df.dtypes["branch"] != np.int64:
        new_col = df["branch"].replace(["P", "Q", "R"], [-1, 0, 1])
        df["branch"] = new_col

    try:
        assert df.dtypes["branch"] == np.int64
    except AssertionError:
        warn(
            message=(
                f"Expected branch data type to be 'int64', "
                f"got '{df.dtypes['branch']}' instead."
                f"This warning is safe to ignore although it is going to "
                f"reduce the performance of your calculations. "
                f"For further details, see:"
                f"https://github.com/radis/radis/issues/65"
            ),
            # category=PerformanceWarning,
        )


def drop_object_format_columns(df, verbose=True):
    """Remove 'object' columns in a pandas DataFrame.
    They are not useful to us at this time, and they slow down all
    operations (as they are converted to 'object' in pandas DataFrame).
    If you want to keep them, better convert them to some numeric values
    """

    objects = [k for k, v in df.dtypes.items() if v == object]
    for k in objects:
        del df[k]
    if verbose >= 2 and len(objects) > 0:
        print(
            (
                "The following columns had the `object` format and were removed: {0}".format(
                    objects
                )
            )
        )
    return df


def format_paths(s):
    """escape all special characters."""
    if s is not None:
        s = str(s).replace("\\", "/")
    return s


drop_all_but_these = ["id", "iso", "wav", "int", "A", "airbrd", "selbrd", "Tdpair",
                      "Tdpsel", "Pshft", "El", "gp"]


def columns_list_to_load(load_columns_type):
    # Which columns to load
    if load_columns_type == "equilibrium":
        columns = list(drop_all_but_these)
    # elif load_columns_type == "noneq":
    #     columns = list(set(drop_all_but_these) | set(required_non_eq))
    # elif load_columns_type == "diluent":
    #     columns = list(broadening_coeff)
    else:
        raise ValueError(
            f"Expected a list or 'all' for `load_columns`, got `load_columns={load_columns_type}"
        )
    return columns


def _get_isotope_list(isotope, df=None):
    """Returns list of isotopes for given molecule Parse the Input
    conditions (fast). If a line database is given, parse the line database
    instead (slow)

    Parameters
    ----------
    molecule: str
        molecule
    df: pandas DataFrame, or ``None``
        line database to parse. Default ``None``
    """

    # if molecule is not None and self.input.molecule != molecule:
    #     raise ValueError(
    #         "Expected molecule is {0} according to the inputs, but got {1} ".format(
    #             self.input.molecule, molecule
    #         )
    #         + "in line database. Check your `molecule=` parameter, or your "
    #         + "line database."
    #     )

    if df is None:
        isotope_list = isotope.split(",")
    else:  # get all isotopes in line database
        isotope_list = set(df.iso)

    return [int(k) for k in isotope_list]


def nm2cm(wl_nm):
    """(vacuum) nm to cm-1."""
    return 1 / wl_nm * 1e9 / 100


def nm_air2cm(wl_nm_air):
    """(air) nm to cm-1.

    References
    ----------

    :func:`~radis.phys.air.air2vacuum'
    """
    return nm2cm(air2vacuum(wl_nm_air))


def load_hitran(
        wavenum_min,
        wavenum_max,
        molecule,
        isotope,
        source="hitran",
        database="full",
        parfunc=None,
        parfuncfmt="hapi",
        # levels=None,
        # levelsfmt="radis",
        load_energies=False,
        include_neighbouring_lines=False,
        # parse_local_global_quanta=True,
        drop_non_numeric=True,
        db_use_cached=True,
        # lvl_use_cached=True,
        # memory_mapping_engine="default",
        load_columns="equilibrium",
        parallel=True,
        extra_params=None,
        drop_column=True,
):
    print("start to load data from HITRAN")

    """Fetch the latest files from [HITRAN-2020]_, [HITEMP-2010]_ (or newer),
    [ExoMol-2020]_  or [GEISA-2020] , and store them locally in memory-mapping
    formats for extremelly fast access.
    Parameters
    ----------
    source: ``'hitran'``, ``'hitemp'``, ``'exomol'``, ``'geisa'``
        which database to use.
    database: ``'full'``, ``'range'``, name of an ExoMol database, or ``'default'``
        if fetching from HITRAN, ``'full'`` download the full database and register
        it, ``'range'`` download only the lines in the range of the molecule.
        .. note::
            ``'range'`` will be faster, but will require a new download each time
            you'll change the range. ``'full'`` is slower and takes more memory, but
            will be downloaded only once.
        Default is ``'full'``.
        If fetching from HITEMP, only ``'full'`` is available.
        if fetching from ''`exomol`'', use this parameter to choose which database
        to use. Keep ``'default'`` to use the recommended one. See all available databases
        with :py:func:`radis.io.exomol.get_exomol_database_list`
        By default, databases are download in ``~/.radisdb``.
        Can be changed in ``radis.config["DEFAULT_DOWNLOAD_PATH"]`` or in
        ``~/radis.json`` config file
    Other Parameters
    ----------------
    parfuncfmt: ``'cdsd'``, ``'hapi'``, ``'exomol'``, or any of :data:`~radis.lbl.loader.KNOWN_PARFUNCFORMAT`
        format to read tabulated partition function file. If ``'hapi'``, then
        [HAPI]_ (HITRAN Python interface) is used to retrieve [TIPS-2020]_
        tabulated partition functions.
        If ``'exomol'`` then partition functions are downloaded from ExoMol.
        Default ``'hapi'``.
    parfunc: filename or None
        path to a tabulated partition function file to use.
    levels: dict of str or None
        path to energy levels (needed for non-eq calculations). Format::
            {1:path_to_levels_iso_1, 3:path_to_levels_iso3}.
        Default ``None``
    levelsfmt: ``'cdsd-pc'``, ``'radis'`` (or any of :data:`~radis.lbl.loader.KNOWN_LVLFORMAT`) or ``None``
        how to read the previous file. Known formats: (see :data:`~radis.lbl.loader.KNOWN_LVLFORMAT`).
        If ``radis``, energies are calculated using the diatomic constants in radis.db database
        if available for given molecule. Look up references there.
        If ``None``, non equilibrium calculations are not possible. Default ``'radis'``.
    load_energies: boolean
        if ``False``, dont load energy levels. This means that nonequilibrium
        spectra cannot be calculated, but it saves some memory. Default ``False``
    include_neighbouring_lines: bool
        if ``True``, includes off-range, neighbouring lines that contribute
        because of lineshape broadening. The ``neighbour_lines``
        parameter is used to determine the limit. Default ``True``.
    drop_non_numeric: boolean
        if ``True``, non numeric columns are dropped. This improves performances,
        but make sure all the columns you need are converted to numeric formats
        before hand. Default ``True``. Note that if a cache file is loaded it
        will be left untouched.
    db_use_cached: bool, or ``'regen'``
        use cached
    parallel: bool
        if ``True``, uses joblib.parallel to load database with multiple processes
        (works only for HITEMP files)
    load_columns: list, ``'all'``, ``'equilibrium'``, ``'noneq'``, ``diluent``,
        columns names to load.
        If ``'equilibrium'``, only load the columns required for equilibrium
        calculations. If ``'noneq'``, also load the columns required for
        non-LTE calculations. See :data:`~radis.lbl.loader.drop_all_but_these`.
        If ``'all'``, load everything. Note that for performances, it is
        better to load only certain columsn rather than loading them all
        and dropping them with ``drop_columns``.
        If ``diluent`` then all additional columns required for calculating spectrum
        in that diluent is loaded.
        Default ``'equilibrium'``.
        .. warning::
            if using ``'equilibrium'``, not all parameters will be available
            for a Spectrum :py:func:`~radis.spectrum.spectrum.Spectrum.line_survey`.
            If you are calculating equilibrium (LTE) spectra, it is recommended to
            use ``'equilibrium'``. If you are calculating non-LTE spectra, it is
            recommended to use ``'noneq'``.
    Notes
    -----
    HITRAN is fetched with Astroquery [1]_ or [HAPI]_,  and HITEMP with
    :py:func:`~radis.io.hitemp.fetch_hitemp`
    HITEMP files are generated in a ~/.radisdb database.
    See Also
    --------
    :meth:`~radis.lbl.loader.DatabankLoader.load_databank`,
    :meth:`~radis.lbl.loader.DatabankLoader.init_databank`
    References
    ----------
    .. [1] `Astroquery <https://astroquery.readthedocs.io>`_
    """
    # To keep things simple, only consider loading data from Hitran for now
    assert (source == "hitran")
    dbformat = "hitran"

    local_databases = data_path  # config["DEFAULT_DOWNLOAD_PATH"], but HARDCODED for my local machine!!

    # Get inputs
    if not molecule:
        raise ValueError("Please define `molecule=` so the database can be fetched")

    # if include_neighbouring_lines:
    #     wavenum_min = self.params.wavenum_min_calc
    #     wavenum_max = self.params.wavenum_max_calc
    # else:
    #     wavenum_min = self.input.wavenum_min
    #     wavenum_max = self.input.wavenum_max

    # Let's store all params so they can be parsed by "get_conditions()"
    # and saved in output spectra information
    # self.params.dbformat = dbformat
    # self.misc.load_energies = load_energies
    # self.levels = levels
    # self.params.levelsfmt = levelsfmt
    # self.params.parfuncpath = format_paths(parfunc)
    # self.params.parfuncfmt = parfuncfmt
    # self.params.db_use_cached = db_use_cached

    # Which columns to load
    columns = []
    if "all" in load_columns:
        columns = None  # see fetch_hitemp, fetch_hitran, etc.
    elif isinstance(load_columns, str) and load_columns in ["equilibrium", "noneq"]:
        columns = columns_list_to_load(load_columns)
    elif load_columns == "diluent":
        raise ValueError(
            "Please use diluent along with 'equilibrium' or 'noneq' in a list like ['diluent','noneq']"
        )

    elif isinstance(load_columns, list) and "all" not in load_columns:
        for load_columns_type in load_columns:
            if load_columns_type in ["equilibrium", "noneq", "diluent"]:
                for col in columns_list_to_load(load_columns_type):
                    columns.append(col)
            # elif load_columns_type in list(
            #         set(drop_all_but_these)
            #         | set(required_non_eq)
            #         | set(broadening_coeff)
            # ):
            #     columns.append(load_columns_type)
            else:
                raise ValueError("invalid column name provided")
        columns = list(set(columns))

    # %% Init Line database
    # ---------------------

    if source == "hitran":
        if database == "full":
            # if memory_mapping_engine == "auto":
            #     memory_mapping_engine = "vaex"

            if isotope == "all":
                isotope_list = None
            else:
                isotope_list = ",".join([str(k) for k in _get_isotope_list(isotope)])

            df, local_paths = fetch_hitran(
                molecule,
                isotope=isotope_list,
                local_databases=join(local_databases, "hitran"),
                load_wavenum_min=wavenum_min,
                load_wavenum_max=wavenum_max,
                columns=columns,
                cache=db_use_cached,
                # verbose=self.verbose,
                return_local_path=True,
                # engine=memory_mapping_engine,
                parallel=parallel,
                extra_params=extra_params,
            )
            # self.params.dbpath = ",".join(local_paths)

            # ... explicitely write all isotopes based on isotopes found in the database
            if isotope == "all":
                isotope = ",".join(
                    [str(k) for k in _get_isotope_list(isotope, df=df)]
                )

        elif database == "range":

            # Query one isotope at a time
            if isotope == "all":
                raise ValueError(
                    "Please define isotope explicitely (cannot use 'all' with fetch_databank('hitran'))"
                )
            isotope_list = _get_isotope_list(isotope)

            frames = []  # lines for all isotopes
            for iso in isotope_list:
                df = None  # fetch_astroquery(
                #     molecule, iso, wavenum_min, wavenum_max, verbose=self.verbose
                # )
                if len(df) > 0:
                    frames.append(df)
                else:
                    print(
                        "No line for isotope n°{}".format(iso),
                        "EmptyDatabaseWarning",
                    )

            # Merge
            if frames == []:
                raise ValueError(
                    f"{molecule} has no lines on range "
                    + "{0:.2f}-{1:.2f} cm-1".format(wavenum_min, wavenum_max)
                )
            if len(frames) > 1:
                # Note @dev : may be faster/less memory hungry to keep lines separated for each isotope. TODO : test both versions
                for df in frames:
                    assert "iso" in df.columns
                df = pd.concat(frames, ignore_index=True)  # reindex
            else:
                df = frames[0]
            # self.params.dbpath = "fetched from hitran"
        else:
            raise ValueError(
                f"Got `database={database}`. When fetching HITRAN, choose `database='full'` to download all database (once for all) or `database='range'` to download only the lines in the current range."
            )
    else:
        raise NotImplementedError("source: {0}".format(source))

    if len(df) == 0:
        raise ValueError(
            f"{molecule} has no lines on range "
            + "{0:.2f}-{1:.2f} cm-1".format(wavenum_min, wavenum_max)
        )

    # Always sort line database by wavenumber (required to SPARSE_WAVERANGE mode)
    df.sort_values("wav", ignore_index=True, inplace=True)

    # Post-processing of the line database
    # (note : this is now done in 'fetch_hitemp' before saving to the disk)
    # spectroscopic quantum numbers will be needed for nonequilibrium calculations, and line survey.
    # if parse_local_global_quanta and "locu" in df and source != "geisa":
    #     df = parse_local_quanta(df, molecule, verbose=self.verbose)
    # if (
    #         parse_local_global_quanta and "globu" in df and source != "geisa"
    # ):  # spectroscopic quantum numbers will be needed for nonequilibrium calculations :
    #     df = parse_global_quanta(df, molecule, verbose=self.verbose)

    # Remove non numerical attributes
    if drop_non_numeric:
        if "branch" in df:
            replace_PQR_with_m101(df)
        df = drop_object_format_columns(df)

    # self.df0 = df  # type : pd.DataFrame
    # self.misc.total_lines = len(df)  # will be stored in Spectrum metadata

    # %% Init Partition functions (with energies)
    # ------------

    # if parfuncfmt == "exomol":
    #     self._init_equilibrium_partition_functions(
    #         parfunc,
    #         parfuncfmt,
    #         # predefined_partition_functions=partition_function_exomol,
    #     )
    # else:
    #     self._init_equilibrium_partition_functions(parfunc, parfuncfmt)

    # If energy levels are given, initialize the partition function calculator
    # (necessary for non-equilibrium). If levelsfmt == 'radis' then energies
    # are calculated ab initio from radis internal species database constants
    # if load_energies:
    #     try:
    #         self._init_rovibrational_energies(levels, levelsfmt)
    #     except KeyError as err:
    #         print(err)
    #         raise KeyError(
    #             "Error while fetching rovibrational energies for "
    #             + "{0}, iso={1} in RADIS built-in spectroscopic ".format(
    #                 molecule, isotope
    #             )
    #             + "constants (see details above). If you only need "
    #             + "equilibrium spectra, try using 'load_energies=False' "
    #             + "in fetch_databank"
    #         )
    #
    # TODO: MIGHT cause some issue if not removing id & isotope column
    remove_unecessary_columns(df)

    return df


def remove_unecessary_columns(df):
    """Remove unecessary columns and add values as attributes

    Returns
    -------
    None: DataFrame updated inplace
    """

    # Discard molecule column if unique
    if "id" in df.columns:
        id_set = df.id.unique()
        if len(id_set) != 1:  # only 1 molecule supported ftm
            raise NotImplementedError(
                "Only 1 molecule at a time is currently supported "
                + "in SpectrumFactory. Use radis.calc_spectrum, which "
                + "calculates them independently then use MergeSlabs"
            )

        df.drop("id", axis=1, inplace=True)
        # df_metadata.append("id")
        df.attrs["id"] = id_set[0]
    else:
        assert "id" in df.attrs or "molecule" in df.attrs

    if "iso" in df.columns:
        isotope_set = df.iso.unique()

        if len(isotope_set) == 1:
            df.drop("iso", axis=1, inplace=True)
            # df_metadata.append("iso")
            df.attrs["iso"] = isotope_set[0]
    else:
        assert "iso" in df.attrs


if __name__ == "__main__":
    load_hitran(wavenum_min=1900,
                wavenum_max=2300,
                molecule='CO',
                isotope='1,2,3')
