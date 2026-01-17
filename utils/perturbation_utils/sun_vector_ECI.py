from astropy.time import Time
from astropy.coordinates import get_sun, GCRS
import astropy.units as u
import numpy as np

def get_sun_vector_eci(epoch):
    """
    Compute the Sun position in ECI (GCRS) frame at a given epoch.

    Parameters
    ----------
    epoch : datetime or str
        UTC epoch (e.g., datetime object or ISO string)

    Returns
    -------
    r_sun_eci : np.ndarray
        Sun position vector in ECI [km]
    """
    # Convert to Astropy Time
    t = Time(epoch, scale='utc')

    # Get Sun position in ICRS
    sun_icrs = get_sun(t)

    # Convert to GCRS (ECI) frame
    sun_gcrs = sun_icrs.transform_to(GCRS(obstime=t))

    # Extract cartesian coordinates in meters, convert to km
    r_sun_eci = sun_gcrs.cartesian.xyz.to(u.km).value # type: ignore

    return np.array(r_sun_eci)

# # ---------------- Example usage ----------------
# if __name__ == "__main__":
#     from datetime import datetime

#     epoch = datetime.utcnow()
#     print("Current UTC time:", epoch)
#     r_sun = get_sun_vector_eci(epoch)

#     print("Sun position in ECI [km]:", r_sun)
#     print("Sun distance [AU]:", np.linalg.norm(r_sun)/149597870.7)  # approx AU
