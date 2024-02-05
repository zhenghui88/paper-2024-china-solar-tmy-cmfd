from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr

# %%
# Inputs and Outputs
DATAROOT = Path("../data")
DAYOUTFILE = DATAROOT.joinpath("cmfd.day.nc")
DAYOUTFILE_PARTS = [
    DATAROOT.joinpath("cmfd.day.1951-1979.nc"),
    DATAROOT.joinpath("cmfd.day.1980-1999.nc"),
    DATAROOT.joinpath("cmfd.day.2000-2020.nc"),
]
MONOUTFILE = DATAROOT.joinpath("cmfd.mon.nc")
YEAOUTFILE = DATAROOT.joinpath("cmfd.yea.nc")

# variables
VARS = {"Dew", "SRad", "Temp", "Wind"}
MAXVARS = {"Dew", "Temp", "Wind"}
MINVARS = {"Dew", "Temp"}

REFDATETIME = datetime(2000, 1, 1)


def create_empty_file_with_headers(OUTFILE, lat, lon, vars, maxvars, minvars):
    REFDATETIME = datetime(2000, 1, 1)
    with h5netcdf.File(OUTFILE, "w", decode_vlen_strings=False) as f:
        f.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
        f.create_variable("time", ("time",), int)
        f["time"].attrs.update(
            {
                "standard_name": np.bytes_("time", "ascii"),
                "units": np.bytes_(
                    f"seconds since {REFDATETIME:%Y-%m-%dT%H:%M:%S}", "ascii"
                ),
            }
        )
        f.create_variable("lat", ("lat",), float, data=lat)
        f["lat"].attrs.update(
            {
                "standard_name": np.bytes_("latitude", "ascii"),
                "units": np.bytes_("degrees_north", "ascii"),
            }
        )
        f.create_variable("lon", ("lon",), float, data=lon)
        f["lon"].attrs.update(
            {
                "standard_name": np.bytes_("longitude", "ascii"),
                "units": np.bytes_("degrees_east", "ascii"),
            }
        )
        for var in vars:
            f.create_variable(
                var.lower(), ("time", "lat", "lon"), "f4", fillvalue=np.nan
            )
        f["dew"].attrs.update(
            {
                "standard_name": np.bytes_("dew_point_temperature", "ascii"),
                "units": np.bytes_("K", "ascii"),
            }
        )
        f["srad"].attrs.update(
            {
                "standard_name": np.bytes_(
                    "surface_downwelling_shortwave_flux_in_air", "ascii"
                ),
                "units": np.bytes_("W m-2", "ascii"),
            }
        )
        f["temp"].attrs.update(
            {
                "standard_name": np.bytes_("air_temperature", "ascii"),
                "units": np.bytes_("K", "ascii"),
            }
        )
        f["wind"].attrs.update(
            {
                "standard_name": np.bytes_("wind_speed", "ascii"),
                "units": np.bytes_("m s-1", "ascii"),
            }
        )
        for var in maxvars:
            f.create_variable(
                f"{var.lower()}_max", ("time", "lat", "lon"), "f4", fillvalue=np.nan
            )
        f["dew_max"].attrs.update(
            {
                "standard_name": np.bytes_("dew_point_temperature", "ascii"),
                "long_name": np.bytes_("daily maximum dew point temperature", "ascii"),
                "units": np.bytes_("K", "ascii"),
            }
        )
        f["temp_max"].attrs.update(
            {
                "standard_name": np.bytes_("air_temperature", "ascii"),
                "long_name": np.bytes_("daily maximum air temperature", "ascii"),
                "units": np.bytes_("K", "ascii"),
            }
        )
        f["wind_max"].attrs.update(
            {
                "standard_name": np.bytes_("wind_speed", "ascii"),
                "long_name": np.bytes_("daily maximum wind speed", "ascii"),
                "units": np.bytes_("m s-1", "ascii"),
            }
        )
        for var in minvars:
            f.create_variable(
                f"{var.lower()}_min", ("time", "lat", "lon"), "f4", fillvalue=np.nan
            )
        f["dew_min"].attrs.update(
            {
                "standard_name": np.bytes_("dew_point_temperature", "ascii"),
                "long_name": np.bytes_("daily minimum dew point temperature", "ascii"),
                "units": np.bytes_("K", "ascii"),
            }
        )
        f["temp_min"].attrs.update(
            {
                "standard_name": np.bytes_("air_temperature", "ascii"),
                "long_name": np.bytes_("daily minimum air temperature", "ascii"),
                "units": np.bytes_("K", "ascii"),
            }
        )


# %%
with h5netcdf.File(
    DAYOUTFILE if DAYOUTFILE.is_file() else DAYOUTFILE_PARTS[0], "r"
) as f:
    lat = f["lat"][:]
    lon = f["lon"][:]

# %%
# merge daily files


def merge_daily_files(dayoutfile: Path, dayoutfile_parts: Sequence[Path]):
    if not all((f.is_file() for f in dayoutfile_parts)):
        return

    dayoutfile.unlink(missing_ok=True)
    create_empty_file_with_headers(
        dayoutfile,
        lat,
        lon,
        VARS,
        MAXVARS,
        MINVARS,
    )

    with h5netcdf.File(dayoutfile, "a") as fo:
        for ncfile in dayoutfile_parts:
            with h5netcdf.File(ncfile) as fi:
                for ii in range(fi.dimensions["time"].size):
                    fo.resize_dimension("time", fo.dimensions["time"].size + 1)
                    for var in fi.variables.keys():
                        if var == "time":
                            ctime = datetime(1900, 1, 1) + timedelta(
                                days=fi.variables[var][ii]
                            )
                            fo.variables[var][-1] = (
                                ctime - REFDATETIME
                            ).total_seconds()
                        elif var in ("lat", "lon"):
                            continue
                        else:
                            fo.variables[var][-1, :, :] = fi[var][ii, :, :]


merge_daily_files(DAYOUTFILE, DAYOUTFILE_PARTS)

# %%
# monthly statistics
MONOUTFILE.unlink(missing_ok=True)
create_empty_file_with_headers(
    MONOUTFILE,
    lat,
    lon,
    VARS,
    MAXVARS,
    MINVARS,
)

# for cday in dayrange((datetime(2010, 1, 19), datetime(2010, 2, 1))):
with xr.open_dataset(DAYOUTFILE) as ds, h5netcdf.File(MONOUTFILE, "a") as fo:
    cmon = datetime(1951, 1, 1)
    while cmon < datetime(2021, 1, 1):
        print(cmon, flush=True)
        if cmon.month == 12:
            nmon = datetime(cmon.year + 1, 1, 1)
        else:
            nmon = datetime(cmon.year, cmon.month + 1, 1)
        fo.resize_dimension("time", fo.dimensions["time"].size + 1)
        fo.variables["time"][-1] = (cmon - REFDATETIME).total_seconds()
        for var in VARS:
            fo.variables[var.lower()][-1, ...] = (
                ds[var.lower()].sel(time=f"{cmon:%Y-%m}").mean(dim="time").to_numpy()
            )
        for var in MAXVARS:
            fo.variables[f"{var.lower()}_max"][-1, ...] = (
                ds[var.lower() + "_max"]
                .sel(time=f"{cmon:%Y-%m}")
                .max(dim="time")
                .to_numpy()
            )
        for var in MINVARS:
            fo.variables[f"{var.lower()}_min"][-1, ...] = (
                ds[var.lower() + "_min"]
                .sel(time=f"{cmon:%Y-%m}")
                .min(dim="time")
                .to_numpy()
            )
        fo.sync()
        cmon = nmon
