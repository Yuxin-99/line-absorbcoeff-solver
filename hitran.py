from os.path import abspath, expanduser, join
from radis.api.hitranapi import HITRANDatabaseManager


def fetch_hitran(
    molecule,
    extra_params=None,
    local_databases=None,
    databank_name="HITRAN-{molecule}",
    isotope=None,
    load_wavenum_min=None,
    load_wavenum_max=None,
    columns=None,
    cache=True,
    verbose=True,
    clean_cache_files=True,
    return_local_path=True,
    engine="default",
    output="pandas",
    parallel=True,
    parse_quanta=False,
):
    """Download all HITRAN lines from HITRAN website. Unzip and build a HDF5 file directly.
    Returns a Pandas DataFrame containing all lines.
    Parameters
    ----------
    molecule: str
        one specific molecule name, listed in HITRAN molecule metadata.
        See https://hitran.org/docs/molec-meta/
        Example: "H2O", "CO2", etc.
    local_databases: str
        where to create the RADIS HDF5 files. Default ``"~/.radisdb/hitran"``.
        Can be changed in ``radis.config["DEFAULT_DOWNLOAD_PATH"]`` or in ~/radis.json config file
    databank_name: str
        name of the databank in RADIS :ref:`Configuration file <label_lbl_config_file>`
        Default ``"HITRAN-{molecule}"``
    isotope: str
        load only certain isotopes : ``'2'``, ``'1,2'``, etc. If ``None``, loads
        everything. Default ``None``.
    load_wavenum_min, load_wavenum_max: float (cm-1)
        load only specific wavenumbers.
    columns: list of str
        list of columns to load. If ``None``, returns all columns in the file.
    extra_params: 'all' or None
        Downloads all additional columns available in the HAPI database for the molecule including
        parameters like `gamma_co2`, `n_co2` that are required to calculate spectrum in co2 diluent.
        For eg:
        ::
            from radis.io.hitran import fetch_hitran
            df = fetch_hitran('CO', extra_params='all', cache='regen') # cache='regen' to regenerate new database with additional columns
    Other Parameters
    ----------------
    cache: ``True``, ``False``, ``'regen'`` or ``'force'``
        if ``True``, use existing HDF5 file. If ``False`` or ``'regen'``, rebuild it.
        If ``'force'``, raise an error if cache file cannot be used (useful for debugging).
        Default ``True``.
    verbose: bool
    clean_cache_files: bool
        if ``True`` clean downloaded cache files after HDF5 are created.
    return_local_path: bool
        if ``True``, also returns the path of the local database file.
    engine: 'pytables', 'vaex', 'default'
        which HDF5 library to use. If 'default' use the value from ~/radis.json
    output: 'pandas', 'vaex', 'jax'
        format of the output DataFrame. If ``'jax'``, returns a dictionary of
        jax arrays. If ``'vaex'``, output is a :py:class:`vaex.dataframe.DataFrameLocal`
        .. note::
            Vaex DataFrames are memory-mapped. They do not take any space in RAM
            and are extremelly useful to deal with the largest databases.
    parallel: bool
        if ``True``, uses joblib.parallel to load database with multiple processes
    parse_quanta: bool
        if ``True``, parse local & global quanta (required to identify lines
        for non-LTE calculations ; but sometimes lines are not labelled.)
    Returns
    -------
    df: pd.DataFrame
        Line list
        A HDF5 file is also created in ``local_databases`` and referenced
        in the :ref:`RADIS config file <label_lbl_config_file>` with name
        ``databank_name``
    local_path: str
        path of local database file if ``return_local_path``
    Examples
    --------
    ::
        from radis.io.hitran import fetch_hitran
        df = fetch_hitran("CO")
        print(df.columns)
        >>> Index(['id', 'iso', 'wav', 'int', 'A', 'airbrd', 'selbrd', 'El', 'Tdpair',
            'Pshft', 'gp', 'gpp', 'branch', 'jl', 'vu', 'vl'],
            dtype='object')
    .. minigallery:: radis.fetch_hitran
    Notes
    -----
    if using ``load_only_wavenum_above/below`` or ``isotope``, the whole
    database is anyway downloaded and uncompressed to ``local_databases``
    fast access .HDF5 files (which will take a long time on first call). Only
    the expected wavenumber range & isotopes are returned. The .HFD5 parsing uses
    :py:func:`~radis.io.hdf5.hdf2df`
    See Also
    --------
    :py:func:`~radis.io.hitemp.fetch_hitemp`, :py:func:`~radis.io.exomol.fetch_exomol`,
    :py:func:`~radis.io.geisa.fetch_geisa`,
    :py:func:`~radis.api.hdf5.hdf2df`, :py:meth:`~radis.lbl.loader.DatabankLoader.fetch_databank`
    """

    if r"{molecule}" in databank_name:
        databank_name = databank_name.format(**{"molecule": molecule})

    if local_databases is None:
        import radis

        local_databases = join(radis.config["DEFAULT_DOWNLOAD_PATH"], "hitran")
    local_databases = abspath(local_databases.replace("~", expanduser("~")))

    ldb = HITRANDatabaseManager(
        databank_name,
        molecule=molecule,
        local_databases=local_databases,
        engine=engine,
        extra_params=extra_params,
        verbose=verbose,
        parallel=parallel,
    )

    # Get expected local files for this database:
    local_file = ldb.get_filenames()

    # Delete files if needed:
    if cache == "regen":
        ldb.remove_local_files(local_file)
    else:
        # Raising AccuracyWarning if local_file exists and doesn't have extra columns in it
        if ldb.get_existing_files(local_file) and extra_params == "all":
            columns = ldb.get_columns(local_file[0])
            extra_columns = ["y_", "gamma_", "n_"]
            found = False
            for key in extra_columns:
                for column_name in columns:
                    if key in column_name:
                        found = True
                        break

            # if not found:
            #     import warnings
            #
            #     warnings.warn(
            #         AccuracyWarning(
            #             "All columns are not downloaded currently, please use cache = 'regen' and extra_params='all' to download all columns."
            #         )
            #     )

    ldb.check_deprecated_files(
        ldb.get_existing_files(local_file),
        auto_remove=True if cache != "force" else False,
    )

    # Download files
    download_files = ldb.get_missing_files(local_file)
    if download_files:
        ldb.download_and_parse(download_files, cache=cache, parse_quanta=parse_quanta)

    # Register
    if not ldb.is_registered():
        ldb.register()

    if len(download_files) > 0 and clean_cache_files:
        ldb.clean_download_files()

    # Load and return
    df = ldb.load(
        local_file,
        columns=columns,
        within=[("iso", isotope)] if isotope is not None else [],
        # for relevant files, get only the right range :
        lower_bound=[("wav", load_wavenum_min)] if load_wavenum_min is not None else [],
        upper_bound=[("wav", load_wavenum_max)] if load_wavenum_max is not None else [],
        output=output,
    )

    return (df, local_file) if return_local_path else df


# class HITRANDatabaseManager(DatabaseManager):
#     def __init__(
#         self,
#         name,
#         molecule,
#         local_databases,
#         engine="default",
#         extra_params=None,
#         verbose=True,
#         parallel=True,
#     ):
#         super().__init__(
#             name,
#             molecule,
#             local_databases,
#             engine=engine,
#             extra_params=extra_params,
#             verbose=verbose,
#             parallel=parallel,
#         )
#         self.downloadable = True
#         self.base_url = None
#         self.Nlines = None
#         self.wmin = None
#         self.wmax = None
#
#     def get_filenames(self):
#         if self.engine == "vaex":
#             return [join(self.local_databases, f"{self.molecule}.hdf5")]
#         elif self.engine == "pytables":
#             return [join(self.local_databases, f"{self.molecule}.h5")]
#         else:
#             raise NotImplementedError()
#
#     def download_and_parse(self, local_file, cache=True, parse_quanta=True):
#         """Download from HITRAN and parse into ``local_file``.
#         Also add metadata
#         Overwrites :py:meth:`radis.api.dbmanager.DatabaseManager.download_and_parse`
#         which downloads from a list of URL, because here we use [HAPI]_ to
#         download the files.
#         Parameters
#         ----------
#         opener: an opener with an .open() command
#         gfile : file handler. Filename: for info"""
#
#         from hapi import LOCAL_TABLE_CACHE, db_begin, fetch
#
#         from radis.db.classes import get_molecule_identifier
#
#         if isinstance(local_file, list):
#             assert (
#                 len(local_file) == 1
#             )  # fetch_hitran stores all lines of a given molecule in one file
#             local_file = local_file[0]
#
#         wmin = 1
#         wmax = 40000
#
#         def download_all_hitran_isotopes(molecule, directory, extra_params):
#             """Blindly try to download all isotpes 1 - 9 for the given molecule
#             .. warning::
#                 this won't be able to download higher isotopes (ex : isotope 10-11-12 for CO2)
#                 Neglected for the moment, they're irrelevant for most calculations anyway
#             .. Isotope Missing:
#                 When an isotope is missing for a particular molecule then a key error `(molecule_id, isotope_id)
#                 is raised.
#             """
#             directory = abspath(expanduser(directory))
#
#             # create temp folder :
#             from radis.misc.basics import make_folders
#
#             make_folders(*split(directory))
#
#             db_begin(directory)
#             isotope_list = []
#             data_file_list = []
#             header_file_list = []
#             for iso in range(1, 10):
#                 file = f"{molecule}_{iso}"
#                 if exists(join(directory, file + ".data")):
#                     if cache == "regen":
#                         # remove without printing message
#                         os.remove(join(directory, file + ".data"))
#                     else:
#                         from radis.misc.printer import printr
#
#                         printr(
#                             "File already exist: {0}. Deleting it.`".format(
#                                 join(directory, file + ".data")
#                             )
#                         )
#                         os.remove(join(directory, file + ".data"))
#                 try:
#                     if extra_params == "all":
#                         fetch(
#                             file,
#                             get_molecule_identifier(molecule),
#                             iso,
#                             wmin,
#                             wmax,
#                             ParameterGroups=[*PARAMETER_GROUPS_HITRAN],
#                         )
#                     elif extra_params is None:
#                         fetch(file, get_molecule_identifier(molecule), iso, wmin, wmax)
#                     else:
#                         raise ValueError("extra_params can only be 'all' or None ")
#                 except KeyError as err:
#                     list_pattern = ["(", ",", ")"]
#                     import re
#
#                     if (
#                         set(list_pattern).issubset(set(str(err)))
#                         and len(re.findall("\d", str(err))) >= 2
#                         and get_molecule_identifier(molecule)
#                         == int(
#                             re.findall(r"[\w']+", str(err))[0]
#                         )  # The regex are cryptic
#                     ):
#                         # Isotope not defined, go to next isotope
#                         continue
#                     else:
#                         raise KeyError("Error: {0}".format(str(err)))
#                 else:
#                     isotope_list.append(iso)
#                     data_file_list.append(file + ".data")
#                     header_file_list.append(file + ".header")
#             return isotope_list, data_file_list, header_file_list
#
#         molecule = self.molecule
#         wmin_final = 100000
#         wmax_final = -1
#
#         # create database in a subfolder to isolate molecules from one-another
#         # (HAPI doesn't check and may mix molecules --> see failure at https://app.travis-ci.com/github/radis/radis/jobs/548126303#L2676)
#         tempdir = join(self.tempdir, molecule)
#         extra_params = self.extra_params
#
#         # Use HAPI only to download the files, then we'll parse them with RADIS's
#         # parsers, and convert to RADIS's fast HDF5 file formats.
#         isotope_list, data_file_list, header_file_list = download_all_hitran_isotopes(
#             molecule, tempdir, extra_params
#         )
#
#         writer = self.get_datafile_manager()
#
#         # Create HDF5 cache file for all isotopes
#         Nlines = 0
#         for iso, data_file in zip(isotope_list, data_file_list):
#             df = pd.DataFrame(LOCAL_TABLE_CACHE[data_file.split(".")[0]]["data"])
#             df.rename(
#                 columns={
#                     "molec_id": "id",
#                     "local_iso_id": "iso",
#                     "nu": "wav",
#                     "sw": "int",
#                     "a": "A",
#                     "gamma_air": "airbrd",
#                     "gamma_self": "selbrd",
#                     "elower": "El",
#                     "n_air": "Tdpair",
#                     "delta_air": "Pshft",
#                     "global_upper_quanta": "globu",
#                     "global_lower_quanta": "globl",
#                     "local_upper_quanta": "locu",
#                     "local_lower_quanta": "locl",
#                     "gp": "gp",
#                     "gpp": "gpp",
#                 },
#                 inplace=True,
#             )
#             df = post_process_hitran_data(
#                 df,
#                 molecule=molecule,
#                 parse_quanta=parse_quanta,
#             )
#
#             wmin_final = min(wmin_final, df.wav.min())
#             wmax_final = max(wmax_final, df.wav.max())
#             Nlines += len(df)
#
#             writer.write(
#                 local_file, df, append=True
#             )  # create temporary files if required
#
#         # Open all HDF5 cache files and export in a single file with Vaex
#         writer.combine_temp_batch_files(
#             local_file, sort_values="wav"
#         )  # used for vaex mode only
#         # Note: by construction, in Pytables mode the database is not sorted
#         # by 'wav' but by isotope
#
#         self.wmin = wmin_final
#         self.wmax = wmax_final
#
#         # Add metadata
#         from radis import __version__
#
#         writer.add_metadata(
#             local_file,
#             {
#                 "wavenumber_min": self.wmin,
#                 "wavenumber_max": self.wmax,
#                 "download_date": self.get_today(),
#                 "download_url": "downloaded by HAPI, parsed & store with RADIS",
#                 "total_lines": Nlines,
#                 "version": __version__,
#             },
#         )
#
#         # # clean downloaded files  TODO
#         # for file in data_file_list + header_file_list:
#         #     os.remove(join(self.local_databases, "downloads", file))
#
#     def register(self):
#         """register in ~/radis.json"""
#
#         from radis.db import MOLECULES_LIST_NONEQUILIBRIUM
#
#         local_files = self.get_filenames()
#
#         if self.wmin is None or self.wmax is None:
#             print(
#                 "Somehow wmin and wmax was not given for this database. Reading from the files"
#             )
#             ##  fix:
#             # (can happen if database was downloaded & parsed, but registration failed a first time)
#             df_full = self.load(
#                 local_files,
#                 columns=["wav"],
#                 within=[],
#                 lower_bound=[],
#                 upper_bound=[],
#             )
#             self.wmin = df_full.wav.min()
#             self.wmax = df_full.wav.max()
#             print(
#                 f"Somehow wmin and wmax was not given for this database. Read {self.wmin}, {self.wmax} directly from the files"
#             )
#
#         info = f"HITRAN {self.molecule} lines ({self.wmin:.1f}-{self.wmax:.1f} cm-1) with TIPS-2021 (through HAPI) for partition functions"
#
#         dict_entries = {
#             "info": info,
#             "path": local_files,
#             "format": "hdf5-radisdb",
#             "parfuncfmt": "hapi",
#             "wavenumber_min": self.wmin,
#             "wavenumber_max": self.wmax,
#             "download_date": self.get_today(),
#         }
#
#         # Add energy level calculation
#         if self.molecule in MOLECULES_LIST_NONEQUILIBRIUM:
#             dict_entries[
#                 "info"
#             ] += " and RADIS spectroscopic constants for rovibrational energies (nonequilibrium)"
#             dict_entries["levelsfmt"] = "radis"
#
#         super().register(dict_entries)