import pywifi
import pywifi.const as wifi
ssid = wifi.interfaces()[0].scan()
print(ssid)