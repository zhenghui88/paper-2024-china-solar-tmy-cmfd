from datetime import datetime
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
MONOUTFILE = DATAROOT.joinpath("cmfd.mon.nc")

FS10YRFILE = DATAROOT.joinpath("cmfd.fs.2011-2020.nc")
FS30YRFILE = DATAROOT.joinpath("cmfd.fs.1991-2020.nc")
FS20YRFILE = DATAROOT.joinpath("cmfd.fs.2001-2020.nc")
FS40YRFILE = DATAROOT.joinpath("cmfd.fs.1981-2020.nc")
FS50YRFILE = DATAROOT.joinpath("cmfd.fs.1971-2020.nc")

# Outputs
TMY10YRFILE = DATAROOT.joinpath("cmfd.tmy.2011-2020.nc")
MYA10YRFILE = DATAROOT.joinpath("cmfd.mya.2011-2020.nc")

TMY20YRFILE = DATAROOT.joinpath("cmfd.tmy.2001-2020.nc")
MYA20YRFILE = DATAROOT.joinpath("cmfd.mya.2001-2020.nc")

TMY30YRFILE = DATAROOT.joinpath("cmfd.tmy.1991-2020.nc")
MYA30YRFILE = DATAROOT.joinpath("cmfd.mya.1991-2020.nc")

TMY40YRFILE = DATAROOT.joinpath("cmfd.tmy.1981-2020.nc")
MYA40YRFILE = DATAROOT.joinpath("cmfd.mya.1981-2020.nc")

TMY50YRFILE = DATAROOT.joinpath("cmfd.tmy.1971-2020.nc")
MYA50YRFILE = DATAROOT.joinpath("cmfd.mya.1971-2020.nc")

TMY30YRNOWINDFILE = DATAROOT.joinpath("cmfd.tmy.nowind.1991-2020.nc")
TMY30YRNOTEMPFILE = DATAROOT.joinpath("cmfd.tmy.notemp.1991-2020.nc")
TMY30YRNODEWFILE = DATAROOT.joinpath("cmfd.tmy.nodew.1991-2020.nc")
TMY30YRNOMAXMINFILE = DATAROOT.joinpath("cmfd.tmy.nomaxmin.1991-2020.nc")


# %%
def write_solar_year(path: Path, lat, lon, data):
    """
    Write solar year to file
    """
    REFDATETIME = datetime(1900, 1, 1)
    kwargs = {"compression": "gzip", "compression_opts": 9}
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
        f.create_variable("lat", ("lat",), float, data=lat, **kwargs)
        f["lat"].attrs.update(
            {
                "standard_name": np.bytes_("latitude", "ascii"),
                "units": np.bytes_("degrees_north", "ascii"),
            }
        )
        f.create_variable("lon", ("lon",), float, data=lon, **kwargs)
        f["lon"].attrs.update(
            {
                "standard_name": np.bytes_("longitude", "ascii"),
                "units": np.bytes_("degrees_east", "ascii"),
            }
        )
        f.create_variable("time", ("time",), float, **kwargs)
        f["time"].attrs.update(
            {
                "standard_name": np.bytes_("time", "ascii"),
                "calendar": np.bytes_("standard", "ascii"),
                "units": np.bytes_(f"days since {REFDATETIME.isoformat()}", "ascii"),
            }
        )
        f.create_variable("srad", ("time", "lat", "lon"), dtype=np.float32, **kwargs)
        f["srad"].attrs.update(
            {
                "standard_name": np.bytes_(
                    "surface_downwelling_shortwave_flux_in_air", "ascii"
                ),
                "units": np.bytes_("W m-2", "ascii"),
            }
        )
        f.resize_dimension("time", 12)
        f.variables["time"][:] = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        f.variables["srad"][:] = data


# %%
# build TMY
@njit
def build_tmy(
    srad: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    tmyyearindex: np.ndarray[tuple[int, int, int], np.dtype[np.integer]],
):
    nlat, nlon = srad.shape[-2:]
    tmy: ndarray[tuple[int, int, int], dtype[np.float32]] = np.zeros(
        (12, nlat, nlon), "f4"
    )
    for imon in range(12):
        for ilat in range(nlat):
            for ilon in range(nlon):
                tmy[imon, ilat, ilon] = srad[
                    tmyyearindex[imon, ilat, ilon] * 12 + imon, ilat, ilon
                ]
    return tmy


def calculate_tmy_mya(fsfile: Path, monfile: Path, varweights: dict[str, float]):
    fs = xr.open_dataset(fsfile)
    times = [datetime.fromisoformat(np.datetime_as_string(x)) for x in fs.time]
    lat: ndarray[tuple[int], dtype[np.float64]] = np.array(fs.lat, "f8")
    lon: ndarray[tuple[int], dtype[np.float64]] = np.array(fs.lon, "f8")
    fsscore = np.zeros((len(fs.time) // 12, 12, len(fs.lat), len(fs.lon)), "f8")
    for var in varweights:
        fsscore += (
            fs[var].to_numpy().reshape((-1, 12, len(lat), len(lon))) * varweights[var]
        )
    fs.close()
    tmyyearindex = np.argmin(fsscore, axis=0)

    # TMY and MYA
    dsmon = xr.open_dataset(monfile)
    sradtmy = build_tmy(
        dsmon["srad"].sel(time=slice(times[0], times[-1])).to_numpy(), tmyyearindex
    )
    sradmya: ndarray[tuple[int, int, int], dtype[np.float32]] = np.array(
        dsmon["srad"].sel(time=slice(times[0], times[-1])).groupby("time.month").mean(),
        "f4",
    )
    dsmon.close()

    return lat, lon, sradtmy, sradmya


# %%
daysofmonth = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
daysofyear = np.sum(daysofmonth)
month_weights = daysofmonth / daysofyear

# %%
varweights_default = {
    "srad": 12 / 24,
    "temp": 2 / 24,
    "temp_max": 1 / 24,
    "temp_min": 1 / 24,
    "dew": 2 / 24,
    "dew_max": 1 / 24,
    "dew_min": 1 / 24,
    "wind": 2 / 24,
    "wind_max": 2 / 24,
}

varweights_nowind = {
    "srad": 12 / 20,
    "temp": 2 / 20,
    "temp_max": 1 / 20,
    "temp_min": 1 / 20,
    "dew": 2 / 20,
    "dew_max": 1 / 20,
    "dew_min": 1 / 20,
    "wind": 0,
    "wind_max": 0,
}

varweights_nodew = {
    "srad": 12 / 20,
    "temp": 2 / 20,
    "temp_max": 1 / 20,
    "temp_min": 1 / 20,
    "dew": 0,
    "dew_max": 0,
    "dew_min": 0,
    "wind": 2 / 20,
    "wind_max": 2 / 20,
}

varweights_notemp = {
    "srad": 12 / 20,
    "temp": 0,
    "temp_max": 0,
    "temp_min": 0,
    "dew": 2 / 20,
    "dew_max": 1 / 20,
    "dew_min": 1 / 20,
    "wind": 2 / 20,
    "wind_max": 2 / 20,
}

varweights_nomaxmin = {
    "srad": 12 / 24,
    "temp": 4 / 24,
    "temp_max": 0,
    "temp_min": 0,
    "dew": 4 / 24,
    "dew_max": 0,
    "dew_min": 0,
    "wind": 4 / 24,
    "wind_max": 0,
}

# %%
# 10 years

lat, lon, sradtmy10yr, sradmya10yr = calculate_tmy_mya(
    FS10YRFILE, MONOUTFILE, varweights_default
)
write_solar_year(TMY10YRFILE, lat, lon, sradtmy10yr)
write_solar_year(MYA10YRFILE, lat, lon, sradmya10yr)


# %%
# 20 years

lat, lon, sradtmy20yr, sradmya20yr = calculate_tmy_mya(
    FS20YRFILE, MONOUTFILE, varweights_default
)
write_solar_year(TMY20YRFILE, lat, lon, sradtmy20yr)
write_solar_year(MYA20YRFILE, lat, lon, sradmya20yr)


# %%
# 30 years

lat, lon, sradtmy30yr, sradmya30yr = calculate_tmy_mya(
    FS30YRFILE, MONOUTFILE, varweights_default
)
write_solar_year(TMY30YRFILE, lat, lon, sradtmy30yr)
write_solar_year(MYA30YRFILE, lat, lon, sradmya30yr)


# %%
# 40 years
lat, lon, sradtmy40yr, sradmya40yr = calculate_tmy_mya(
    FS40YRFILE, MONOUTFILE, varweights_default
)
write_solar_year(TMY40YRFILE, lat, lon, sradtmy40yr)
write_solar_year(MYA40YRFILE, lat, lon, sradmya40yr)

# %%
# 50 years
lat, lon, sradtmy50yr, sradmya50yr = calculate_tmy_mya(
    FS50YRFILE, MONOUTFILE, varweights_default
)
write_solar_year(TMY50YRFILE, lat, lon, sradtmy50yr)
write_solar_year(MYA50YRFILE, lat, lon, sradmya50yr)

# %%
# 30 years no wind, no temp, no dew, no maxmin
_, _, sradtmy30yr_nowind, _ = calculate_tmy_mya(
    FS30YRFILE, MONOUTFILE, varweights_nowind
)
write_solar_year(TMY30YRNOWINDFILE, lat, lon, sradtmy30yr_nowind)

_, _, sradtmy30yr_notemp, _ = calculate_tmy_mya(
    FS30YRFILE, MONOUTFILE, varweights_notemp
)
write_solar_year(TMY30YRNOTEMPFILE, lat, lon, sradtmy30yr_notemp)

_, _, sradtmy30yr_nodew, _ = calculate_tmy_mya(FS30YRFILE, MONOUTFILE, varweights_nodew)
write_solar_year(TMY30YRNODEWFILE, lat, lon, sradtmy30yr_nodew)

_, _, sradtmy30yr_nomaxmin, _ = calculate_tmy_mya(
    FS30YRFILE, MONOUTFILE, varweights_nomaxmin
)
write_solar_year(TMY30YRNOMAXMINFILE, lat, lon, sradtmy30yr_nomaxmin)
