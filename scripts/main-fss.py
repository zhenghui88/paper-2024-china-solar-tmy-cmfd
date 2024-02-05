# %%
from datetime import datetime, timedelta
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr
from numba import njit
from numpy import dtype, ndarray

# %%
# Inputs and Outputs
# Inputs
DATAROOT = Path("../data")
DAYOUTFILE = DATAROOT.joinpath("cmfd.day.nc")

# Outputs
FSOUTFILES = {
    DATAROOT.joinpath("cmfd.fs.2011-2020.nc"): (
        datetime(2011, 1, 1),
        datetime(2021, 1, 1),
    ),
    DATAROOT.joinpath("cmfd.fs.2001-2020.nc"): (
        datetime(2001, 1, 1),
        datetime(2021, 1, 1),
    ),
    DATAROOT.joinpath("cmfd.fs.1991-2020.nc"): (
        datetime(1991, 1, 1),
        datetime(2021, 1, 1),
    ),
    DATAROOT.joinpath("cmfd.fs.1981-2020.nc"): (
        datetime(1981, 1, 1),
        datetime(2021, 1, 1),
    ),
}

# variables
VARS = {
    "srad",
    "dew",
    "dew_max",
    "dew_min",
    "temp",
    "temp_max",
    "temp_min",
    "wind",
    "wind_max",
}


# %%
# Finkelstein-Schafer statistic
@njit
def finkelstein_schafer_statistic(
    sorted_reference: ndarray[tuple[int, int, int], np.dtype[np.floating]],
    sorted_target: ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> ndarray[tuple[int, int], dtype[np.floating]]:
    out = np.full((sorted_target.shape[0], sorted_target.shape[1]), np.inf, np.float32)
    ntarget = sorted_target.shape[2]
    nreference = sorted_reference.shape[2]
    starget = (np.arange(ntarget, dtype=np.float32) + 0.5) / ntarget
    for ii in range(sorted_target.shape[0]):
        for jj in range(sorted_target.shape[1]):
            if np.any(np.isnan(sorted_target[ii, jj, :])) or np.any(
                np.isnan(sorted_reference[ii, jj, :])
            ):
                continue
            sreference = (
                np.maximum(
                    np.searchsorted(
                        sorted_reference[ii, jj, :],
                        sorted_target[ii, jj, :],
                        side="right",
                    )
                    - 0.5,
                    0.0,
                )
                / nreference
            )
            out[ii, jj] = np.mean(np.abs(starget - sreference))
    return out


# %%
def create_empty_file_with_headers(OUTFILE, time, lat, lon, vars, description):
    REFDATETIME = datetime(2000, 1, 1)
    empty_value = np.full((len(time), len(lat), len(lon)), np.nan, dtype=np.float32)
    with h5netcdf.File(OUTFILE, "w") as f:
        f.attrs.update(
            {
                "description": np.bytes_(
                    description,
                    "ascii",
                )
            }
        )
        f.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
        f.create_variable(
            "time",
            ("time",),
            int,
        )
        f.resize_dimension("time", len(time))
        f.variables["time"][:] = [(t - REFDATETIME).total_seconds() for t in time]
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
                var.lower(),
                ("time", "lat", "lon"),
                "f4",
                fillvalue=np.nan,
                data=empty_value,
            )
        f["dew"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for dew point temperature",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["srad"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for surface downwelling shortwave flux in air",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["temp"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for air temperature", "ascii"
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["wind"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for wind speed", "ascii"
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["dew_max"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for daily maximum dew point temperature",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["temp_max"].attrs.update(
            {
                "standard_name": np.bytes_("air_temperature", "ascii"),
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for daily maximum air temperature",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["wind_max"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for daily maximum wind speed",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["dew_min"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for daily minimum dew point temperature",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )
        f["temp_min"].attrs.update(
            {
                "long_name": np.bytes_(
                    "the Finkelstein Schafer statistic for daily minimum air temperature",
                    "ascii",
                ),
                "units": np.bytes_("1", "ascii"),
            }
        )


# %%


def calculate_write_fss(
    fsoutfile: Path,
    date_range: tuple[datetime, datetime],
    dsday: xr.Dataset,
    vars: set[str],
):
    months = [
        datetime(y, m, 1)
        for y in range(date_range[0].year, date_range[1].year)
        for m in range(1, 13)
    ]
    fsoutfile.unlink(missing_ok=True)
    create_empty_file_with_headers(
        fsoutfile,
        months,
        dsday.lat,
        dsday.lon,
        vars,
        f"The Finkelstein Schafer statistic calculated using the CMFD with reference of {date_range[0].year:04d} to {date_range[1].year-1:04d}",
    )
    for var in vars:
        for iclim, clim in (
            dsday[var]
            .sel(time=slice(date_range[0], date_range[1] - timedelta(seconds=1)))
            .groupby("time.month")
        ):
            climmonth = datetime.fromisoformat(
                np.datetime_as_string(clim.time.values[0])
            ).month
            print(var, climmonth, flush=True)
            climdata = np.sort(np.moveaxis(np.array(clim.to_numpy()), 0, -1), -1)
            for imon, mon in enumerate(months):
                if mon.month != climmonth:
                    continue
                print(mon, flush=True)
                mondata = np.sort(
                    np.moveaxis(
                        np.array(dsday[var].sel(time=f"{mon:%Y-%m}").to_numpy()), 0, -1
                    ),
                    -1,
                )
                with h5netcdf.File(fsoutfile, "a") as f:
                    f.variables[var][imon, :, :] = finkelstein_schafer_statistic(
                        climdata,
                        mondata,
                    )


dsday = xr.open_dataset(DAYOUTFILE)

for fsoutfile, date_range in FSOUTFILES.items():
    calculate_write_fss(fsoutfile, date_range, dsday, VARS)
