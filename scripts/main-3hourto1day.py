# %%
from array import array
from datetime import datetime, timedelta
from functools import lru_cache
from math import exp, isfinite, pow
from pathlib import Path

import h5netcdf
import numpy as np
from numba import njit

# %%
# Inputs and Outputs
DATETIME_START = datetime(1951, 1, 1)
DATETIME_STOP = datetime(2021, 1, 1)

DATAROOT = Path("../data")
CMFD_ROOT = DATAROOT.joinpath("cmfd_v1.7/Data_forcing_03hr_010deg")
DAYOUTFILE = DATAROOT.joinpath("cmfd.day.nc")

# variables used in this study
VARS = {"Dew", "SRad", "Temp", "Wind"}
MAXVARS = {"Dew", "Temp", "Wind"}
MINVARS = {"Dew", "Temp"}


# %%
# the dew point temperature
@njit
def dew_point_temperature(specific_humidity, pressure):
    """calculate the dew point temperature from specific humidity [kg kg-1] and pressure [Pa]"""
    TD_MIN = 273.16 - 75.0
    TD_INIT = 273.16 + 10.0
    ret = np.full(specific_humidity.size, np.nan)
    for ii, (q, p) in enumerate(zip(specific_humidity.flat, pressure.flat)):
        if isfinite(q) and isfinite(p):
            td = TD_INIT
            tdp = td - 1.0
            vp = q * p / (0.622 + 0.378 * q)
            while abs(td - tdp) >= 1.0e-5:
                tdp = td
                svp = exp(34.494 - (4924.99 / (td - 36.06))) / pow(td - 168.16, 1.57)
                dsvp = svp * (4924.99 / (td - 36.06) ** 2 - 1.57 / (td - 168.16))
                td = td - (svp - vp) / dsvp
                if td <= TD_MIN:
                    td = TD_MIN
                    tdp = TD_MIN
            ret[ii] = td
    return np.reshape(ret, specific_humidity.shape)


# %%
# the CMFD class


class CMFD:
    VERSION = "0107"
    TIMESPAN = (datetime(1951, 1, 1), datetime(2021, 1, 1))  # inclusive, exclusive
    TIMESTEP = timedelta(hours=3)
    _DISK_VARS = {"LRad", "Prec", "Pres", "RHum", "SHum", "SRad", "Temp", "Wind"}

    def __init__(
        self,
        root: Path,
    ):
        self.root = root
        self._calc_vars = {"Dew": self._calcvars_dew}

    def timerange(self, timespan: tuple[datetime, datetime] | None = None):
        if timespan is None:
            timespan = CMFD.TIMESPAN
        cdt = timespan[0]
        while cdt < timespan[1]:
            yield cdt
            cdt += CMFD.TIMESTEP

    def get(self, var: str, dt: datetime):
        if (dt < CMFD.TIMESPAN[0]) or (dt >= CMFD.TIMESPAN[1]):
            raise IndexError(f"datetime {dt} is before the first CMFD record")
        if (dt - CMFD.TIMESPAN[0]).total_seconds() % CMFD.TIMESTEP.total_seconds() != 0:
            raise IndexError(f"datetime {dt} is not a multiple of the timestep")
        if var in CMFD._DISK_VARS:
            return self.get_diskvars(var, dt)
        elif var in self._calc_vars:
            return self._calc_vars[var](dt)
        else:
            raise KeyError(f"CMFD variable {var} not found")

    @lru_cache(maxsize=1)
    def _calcvars_dew(self, dt: datetime):
        p = self.get_diskvars("Pres", dt)
        q = self.get_diskvars("SHum", dt)
        return dew_point_temperature(q, p)

    @lru_cache(maxsize=8)
    def get_diskvars(self, var: str, dt: datetime):
        if var not in CMFD._DISK_VARS:
            raise KeyError(f"CMFD variable {var} not found in disk")
        ncfile = self._filepath(var, dt)
        idt = self._filetimeindex(dt)
        with h5netcdf.File(ncfile, "r") as f:
            ctime = datetime(1900, 1, 1) + timedelta(
                hours=float(f.variables["time"][idt])
            )
            assert ctime == dt, f"file time {ctime} does not match {dt}"
            fillvalue = f[var.lower()].attrs["_FillValue"]
            data: np.ndarray = f.variables[var.lower()][idt, :, :]
            data[data == fillvalue] = np.nan
            return data

    def varnames(self):
        return CMFD._DISK_VARS | set(self._calc_vars.keys())

    def lat(self):
        ncfile = self._filepath("LRad", CMFD.TIMESPAN[0])
        with h5netcdf.File(ncfile, "r") as f:
            return array("f", f["lat"][:])

    def lon(self):
        ncfile = self._filepath("LRad", CMFD.TIMESPAN[0])
        with h5netcdf.File(ncfile, "r") as f:
            return array("f", f["lon"][:])

    def _filetimeindex(self, dt: datetime):
        monthbegin = datetime(dt.year, dt.month, 1)
        return (dt - monthbegin) // CMFD.TIMESTEP

    def _filepath(self, var: str, dt: datetime):
        return self.root.joinpath(
            var, f"{var.lower()}_CMFD_V{CMFD.VERSION}_B-01_03hr_010deg_{dt:%Y%m}.nc"
        )


cmfd = CMFD(CMFD_ROOT)


# %%
# the daily mean
def dayrange(timespan: tuple[datetime, datetime]):
    cdt = datetime(timespan[0].year, timespan[0].month, timespan[0].day)
    while cdt < timespan[1]:
        yield cdt
        cdt += timedelta(days=1)


# create an empty file
def create_empty_file_with_headers(OUTFILE, lat, lon, vars, maxvars, minvars):
    REFDATETIME = datetime(2000, 1, 1)
    with h5netcdf.File(OUTFILE, "w", decode_vlen_strings=False) as f:
        f.dimensions = {"time": None, "lat": len(lat), "lon": len(lon)}
        f.create_variable("time", ("time",), float)
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
def append_daily_statistics(
    outfile: Path,
    timespan: tuple[datetime, datetime],
    cmfd: CMFD,
    vars: set[str],
    maxvars: set[str],
    minvars: set[str],
):
    """calculate the daily statistics from cmfd and append to the outfile"""
    REFDATETIME = datetime(2000, 1, 1)
    cumdata = {
        x: np.empty((len(cmfd.lat()), len(cmfd.lon())), dtype="f8") for x in vars
    }
    cumcount = {x: 0 for x in vars}
    maxdata = {
        x: np.empty((len(cmfd.lat()), len(cmfd.lon())), dtype="f4") for x in maxvars
    }
    mindata = {
        x: np.empty((len(cmfd.lat()), len(cmfd.lon())), dtype="f4") for x in minvars
    }

    # for cday in dayrange((datetime(2010, 1, 19), datetime(2010, 2, 1))):
    for cday in dayrange(timespan):
        if cday.day == 1:
            print(cday)
        for var in vars:
            cumcount[var] = 0
            cumdata[var][:] = 0.0
        for var in maxvars:
            maxdata[var][:] = -np.inf
        for var in minvars:
            mindata[var][:] = np.inf
        for cdt in cmfd.timerange((cday, cday + timedelta(days=1))):
            for var in vars:
                cumdata[var][:] += cmfd.get(var, cdt)
                cumcount[var] += 1
            for var in maxvars:
                maxdata[var][:] = np.maximum(maxdata[var], cmfd.get(var, cdt))
            for var in minvars:
                mindata[var][:] = np.minimum(mindata[var], cmfd.get(var, cdt))
        with h5netcdf.File(outfile, "r+") as f:
            f.resize_dimension("time", f.dimensions["time"].size + 1)
            f.variables["time"][-1] = (cday - REFDATETIME).total_seconds()
            for var in vars:
                f.variables[var.lower()][-1, ...] = cumdata[var] / cumcount[var]
            for var in maxvars:
                f.variables[f"{var.lower()}_max"][-1, ...] = maxdata[var]
            for var in minvars:
                f.variables[f"{var.lower()}_min"][-1, ...] = mindata[var]


# %%
# daily output


create_empty_file_with_headers(
    DAYOUTFILE, cmfd.lat(), cmfd.lon(), VARS, MAXVARS, MINVARS
)
append_daily_statistics(
    DAYOUTFILE, (DATETIME_START, DATETIME_STOP), cmfd, VARS, MAXVARS, MINVARS
)
