import datetime
import math
import time
import pytz

def sun_position(latitude, longitude, timestamp, timezone):
    
    # B3 - lat
    B3 = latitude

    # B4 - long
    B4 = longitude

    # B5 - timezone
    tz = pytz.timezone(timezone)
    timestamp_tz = timestamp.astimezone(tz)

    is_dst = timestamp_tz.dst().total_seconds()/3600
    B5 = timestamp_tz.utcoffset().total_seconds()/(60 * 60) - is_dst
    
    # F - Julian Day
    year = timestamp_tz.year
    month = timestamp_tz.month
    day = timestamp_tz.day
    hour = timestamp_tz.hour
    minute = timestamp_tz.minute
    second = timestamp_tz.second
    F = (367 * year) - int((7 * (year + int((month + 9) / 12.0))) / 4.0) + int((275 * month) / 9.0) + day + 1721013.5 + ((second / 60.0 + minute) / 60.0 + hour) / 24.0

    # G - Julian Century
    G=(F-2451545)/36525

    # I - Geom Mean Long Sun (deg)
    I = (280.46646+G*(36000.76983+G*0.0003032))%360

    # J - Geom Mean Anom Sun (deg)
    J=357.52911+G*(35999.05029-0.0001537*G)

    # K - Eccent Earth Orbit
    K=0.016708634-G*(0.000042037+0.0000001267*G)

    # L - Sun Eq of Ctr
    L = math.sin(math.radians(J))*(1.914602-G*(0.004817+0.000014*G))+math.sin(math.radians(2*J))*(0.019993-0.000101*G)+math.sin(math.radians(3*J))*0.000289

    # M - Sun True Long (deg)
    M=I+L

    # N - Sun True Anom (deg)
    N=J+L

    # O - Sun Rad Vector (AUs)
    O=(1.000001018*(1-K*K))/(1+K*math.cos(math.radians(N)))

    # P - Sun App Long (deg)
    P=M-0.00569-0.00478*math.sin(math.radians(125.04-1934.136*G))

    # Q - Mean Obliq Ecliptic (deg)
    Q=23+(26+((21.448-G*(46.815+G*(0.00059-G*0.001813))))/60)/60

    # R - Obliq Corr (deg)
    R=Q+0.00256*math.cos(math.radians(125.04-1934.136*G))

    # S - Sun Rt Ascen (deg)
    S=math.degrees(math.atan2(math.cos(math.radians(P)),math.cos(math.radians(R))*math.sin(math.radians(P))))

    # T - Sun Declin (deg)
    T=math.degrees(math.asin(math.sin(math.radians(R))*math.sin(math.radians(P))))

    # U - var y
    U=math.tan(math.radians(R/2))*math.tan(math.radians(R/2))

    # V - Eq of Time (minutes)
    V=4*math.degrees(U*math.sin(2*math.radians(I))-2*K*math.sin(math.radians(J))+4*K*U*math.sin(math.radians(J))*math.cos(2*math.radians(I))-0.5*U*U*math.sin(4*math.radians(I))-1.25*K*K*math.sin(2*math.radians(J)))

    # W - HA Sunrise (deg)
    W=math.degrees(math.acos(math.cos(math.radians(90.833))/(math.cos(math.radians(B3))*math.cos(math.radians(T)))-math.tan(math.radians(B3))*math.tan(math.radians(T))))

    # X - Solar Noon (LST)
    X=(720-4*B4-V+B5*60)/1440

    # Y - Sunrise Time (LST)
    Y=X-W*4/1440

    # Z - Sunset Time (LST)
    Z=X+W*4/1440

    # AA - Sunlight Duration (minutes)
    AA=8*W

    # AB - True Solar Time (min)
    # =MOD(E45*1440+V45+4*$B$4-60*$B$5,1440)
    msm = (second/60 + minute + hour*60)/1440
    AB=(msm*1440+V+4*B4-60*B5)%1440

    # AC - Hour Angle (deg)
    if AB/4<0:
        AC = AB/4+180
    else:
        AC = AB/4-180

    # AD - Solar Zenith Angle (deg)
    AD=math.degrees(math.acos(math.sin(math.radians(B3))*math.sin(math.radians(T))+math.cos(math.radians(B3))*math.cos(math.radians(T))*math.cos(math.radians(AC))))

    # AE - Solar Elevation Angle (deg)
    AE=90-AD

    # AF - Approx Atmospheric Refraction (deg)
    if AE>85:
        AF = 0
    else:
        if AE>5:
            AF = 58.1/math.tan(math.radians(AE))-0.07/math.pow(math.tan(math.radians(AE)),3)+0.000086/math.pow(math.tan(math.radians(AE)),5)
        else:
            if AE>-0.575:
                AF = 1735+AE*(-518.2+AE*(103.4+AE*(-12.79+AE*0.711)))
            else:
                AF = -20.772/math.tan(math.radians(AE))

    AF = AF/3600

    # AG - Solar Elevation corrected for atm refraction (deg)
    altitude = AE + AF

    # AH - Solar Azimuth Angle (deg cw from N)
    if AC>0:
        azimuth = (math.degrees(math.acos(((math.sin(math.radians(B3))*math.cos(math.radians(AD)))-math.sin(math.radians(T)))/(math.cos(math.radians(B3))*math.sin(math.radians(AD)))))+180)%360
    else:
        azimuth = (540-math.degrees(math.acos(((math.sin(math.radians(B3))*math.cos(math.radians(AD)))-math.sin(math.radians(T)))/(math.cos(math.radians(B3))*math.sin(math.radians(AD))))))%360

    # radiation measured in watts per square meter
    day = timestamp.tz_localize(None).utctimetuple().tm_yday
    is_daytime = (altitude > 0)
    flux = 1160 + (75 * math.sin(2 * math.pi / 365 * (day - 275)))
    optical_depth = 0.174 + (0.035 * math.sin(2 * math.pi / 365 * (day - 100)))
    
    if math.sin(math.radians(altitude)) == 0:
        air_mass_ratio = float("inf")
    else:
        air_mass_ratio = 1 / math.sin(math.radians(altitude))
        
    if (-1 * optical_depth * air_mass_ratio) > 100:
        od_amr = 100
    else:
        od_amr = -1 * optical_depth * air_mass_ratio
        
    radiation = flux * math.exp(od_amr) * is_daytime
    
    return {
        "altitude":altitude, 
        "azimuth":azimuth, 
        "radiation":radiation,
        "sunrise":Y,
        "sunset":Z
    }