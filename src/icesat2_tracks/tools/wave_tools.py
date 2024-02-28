def to_vec(amp, angle, deg = True):

    "from anlge deg to vect"
    import numpy as np

    if deg:
        u, v  = amp * np.cos(angle * np.pi/180), amp * np.sin(angle * np.pi/180)
    else:
        u, v  = amp * np.cos(angle ), amp * np.sin(angle )

    return u,v

def to_deg(u,v, deg = True):

    """
    from vect to angle, amp
    angle is -180 to 180
    this is a different definiton then WW3 [0, 360 ), but (-180, 180] is more convient for the problem
    """
    import numpy as np

    amp = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)# + 2 * np.pi

    # nan_mask = np.isnan(angle)
    # angle= np.where(angle > np.pi, angle - 2 * np.pi , angle)
    # angle= np.where(angle <= - np.pi, angle + 2 * np.pi , angle)
    # angle= np.where(nan_mask, np.nan , angle)

    if deg:
        angle = angle * 180/np.pi
    return amp, angle

def get_ave_amp_angle(amp, angle, deg = True):
    import numpy as np
    u,v =  to_vec(amp, angle , deg= deg)
    # average angle in vector space
    #print(u, v)
    _ , ave_deg = to_deg( np.nanmean(u) , np.nanmean(v), deg = deg)
    _ , std_deg = to_deg( np.nanstd(u) , np.nanstd(v) , deg = deg)

    #average amp in angle space
    ave_amp = np.nanmean(amp)
    std_amp = np.nanstd(amp)

    return ave_amp, ave_deg, std_amp, std_deg
