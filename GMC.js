from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Union, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import logging
import requests
from hijri_converter import Gregorian
from functools import lru_cache

# Configuratie logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constanten
API_KEY = 'YOUR_API_KEY'
EARTH_RADIUS = 6378137  # in meters

@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    timezone_str: str
    elevation: float = 0
    name: str = "Onbekende Locatie"

    def __post_init__(self):
        object.__setattr__(self, '_validate_latitude', self._validate_latitude())
        object.__setattr__(self, '_validate_longitude', self._validate_longitude())
        object.__setattr__(self, '_validate_elevation', self._validate_elevation())

    def _validate_latitude(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Breedtegraad moet tussen -90 en 90 zijn, gekregen: {self.latitude}")

    def _validate_longitude(self):
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Lengtegraad moet tussen -180 en 180 zijn, gekregen: {self.longitude}")

    def _validate_elevation(self):
        if self.elevation < 0:
            raise ValueError(f"Hoogte moet niet negatief zijn, gekregen: {self.elevation}")

class BerekeningsMethode(ABC):
    @property
    @abstractmethod
    def naam(self) -> str:
        pass

    @abstractmethod
    def get_asr_factor(self) -> float:
        pass

    @abstractmethod
    def get_fajr_hoek(self) -> float:
        pass

    @abstractmethod
    def get_isha_hoek(self) -> float:
        pass

class StandaardMethode(BerekeningsMethode):
    @property
    def naam(self) -> str:
        return "Standaard (Shafi'i, Maliki, Hanbali)"

    def get_asr_factor(self) -> float:
        return 1.0

    def get_fajr_hoek(self) -> float:
        return 18.0

    def get_isha_hoek(self) -> float:
        return 17.0

class HanafiMethode(BerekeningsMethode):
    @property
    def naam(self) -> str:
        return "Hanafi"

    def get_asr_factor(self) -> float:
        return 2.0

    def get_fajr_hoek(self) -> float:
        return 18.0

    def get_isha_hoek(self) -> float:
        return 18.0

class AstronomischeBerekeningen:
    """Beheert nauwkeurige astronomische berekeningen voor zonpositie en tijden."""

    def __init__(self, location: Location, datum: datetime):
        self.location = location
        self.datum = datum
        self.julian_dag = self._bereken_julian_dag()

    @lru_cache(maxsize=32)
    def _bereken_julian_dag(self) -> float:
        """Bereken Julian Day vanuit een Gregorian datum."""
        jaar = self.datum.year
        maand = self.datum.month

        if maand <= 2:
            jaar -= 1
            maand += 12

        a = math.floor(jaar / 100)
        b = 2 - a + math.floor(a / 4)

        jd = (math.floor(365.25 * (jaar + 4716)) +
              math.floor(30.6001 * (maand + 1)) +
              self.datum.day + b - 1524.5 +
              (self.datum.hour + self.datum.minute / 60 +
               self.datum.second / 3600) / 24.0)
        return jd

    def _calculate_sun_position(self) -> Dict[str, float]:
        """Bereken declinatie, rechte klimming en tijdvergelijking van de zon."""
        d = self.julian_dag - 2451545.0
        g = 357.529 + 0.98560028 * d
        q = 280.459 + 0.98564736 * d
        g_rad = math.radians(g)

        e = 23.439 - 0.00000036 * d
        e_rad = math.radians(e)

        L = q + 1.915 * math.sin(g_rad) + 0.020 * math.sin(2 * g_rad)
        L_rad = math.radians(L)

        center = 1.915 * math.sin(g_rad) + 0.020 * math.sin(2 * g_rad)
        true_long = q + center

        omega = 125.04 - 1934.136 * d
        lambda_sun = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
        lambda_rad = math.radians(lambda_sun)

        declinatie = math.degrees(math.asin(math.sin(e_rad) * math.sin(lambda_rad)))

        RA = math.degrees(math.atan2(math.cos(e_rad) * math.sin(lambda_rad),
                                     math.cos(lambda_rad)))
        RA = RA / 15  # Omzetten naar uren

        y = math.tan(e_rad / 2) ** 2
        eot = y * math.sin(2 * math.radians(q)) - 2 * self.eccentriciteit() * math.sin(g_rad) + \
              4 * self.eccentriciteit() * y * math.sin(g_rad) * math.cos(2 * math.radians(q)) - \
              0.5 * y * y * math.sin(4 * math.radians(q)) - \
              1.25 * self.eccentriciteit() ** 2 * math.sin(2 * g_rad)
        eot = math.degrees(eot) * 4  # Omzetten naar minuten

        return {
            'declinatie': declinatie,
            'RA': RA,
            'eot': eot
        }

    def eccentriciteit(self) -> float:
        """Bereken de excentriciteit van de aardbaan."""
        T = (self.julian_dag - 2451545.0) / 36525
        return 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T

    def calculate_prayer_times(self, methode: BerekeningsMethode) -> Dict[str, float]:
        """Bereken gebedstijden op basis van astronomische berekeningen."""
        zon_pos = self._calculate_sun_position()
        declinatie = zon_pos['declinatie']
        eot = zon_pos['eot']

        asr_factor = methode.get_asr_factor()

        tijden = {}

        # Bereken zonnoon
        zonnoon = 12 - eot / 60  # GMT tijd
        tijden['dhuhr'] = zonnoon

        # Bereken zonsopkomst en zonsondergang
        alpha = self._get_zon_correctie()
        cos_omega = (math.sin(math.radians(-0.8333 - alpha)) -
                     math.sin(math.radians(self.location.latitude)) *
                     math.sin(math.radians(declinatie))) / \
                    (math.cos(math.radians(self.location.latitude)) *
                     math.cos(math.radians(declinatie)))

        if -1 <= cos_omega <= 1:
            omega = math.degrees(math.acos(cos_omega))
            tijden['sunrise'] = zonnoon - omega / 15
            tijden['sunset'] = zonnoon + omega / 15
            tijden['maghrib'] = tijden['sunset']
        else:
            tijden['sunrise'] = float('nan')
            tijden['sunset'] = float('nan')
            tijden['maghrib'] = float('nan')

        # Bereken Asr tijd
        t = asr_factor + math.tan(math.radians(abs(self.location.latitude - declinatie)))
        asr_hoek = math.degrees(math.atan(1 / t))

        cos_omega_asr = (math.sin(math.radians(90 - asr_hoek)) -
                         math.sin(math.radians(self.location.latitude)) *
                         math.sin(math.radians(declinatie))) / \
                        (math.cos(math.radians(self.location.latitude)) *
                         math.cos(math.radians(declinatie)))

        if -1 <= cos_omega_asr <= 1:
            omega_asr = math.degrees(math.acos(cos_omega_asr))
            tijden['asr'] = zonnoon + omega_asr / 15
        else:
            tijden['asr'] = float('nan')

        # Bereken Fajr en Isha tijden
        fajr_hoek = methode.get_fajr_hoek()
        isha_hoek = methode.get_isha_hoek()

        for gebed, hoek in [('fajr', fajr_hoek), ('isha', isha_hoek)]:
            cos_omega_gebed = (math.sin(math.radians(-hoek)) -
                                math.sin(math.radians(self.location.latitude)) *
                                math.sin(math.radians(declinatie))) / \
                               (math.cos(math.radians(self.location.latitude)) *
                                math.cos(math.radians(declinatie)))

            if -1 <= cos_omega_gebed <= 1:
                omega_gebed = math.degrees(math.acos(cos_omega_gebed))
                if gebed == 'fajr':
                    tijden[gebed] = zonnoon - omega_gebed / 15
                else:
                    tijden[gebed] = zonnoon + omega_gebed / 15
            else:
                tijden[gebed] = float('nan')

        return tijden

    def _get_zon_correctie(self) -> float:
        """Bereken correctie voor zonhoogte op basis van atmosferische refractie en hoogte."""
        refractie = 0.0347  # Refractie bij de horizon in graden
        hoogte = self.location.elevation
        horizon_daling = math.degrees(math.acos(EARTH_RADIUS / (EARTH_RADIUS + hoogte))) if hoogte > 0 else 0
        return refractie + horizon_daling

class MaanDataProvider:
    """Haalt maandata op via de QWeather API."""

    def __init__(self, api_sleutel: str):
        self.api_sleutel = api_sleutel
        self.api_host = 'https://devapi.qweather.com/v7/astronomy/moon'
        self.sessie = requests.Session()

    def haal_maan_data(self, location: Location, datum: datetime) -> Dict:
        """
        Haalt maandata op voor een specifieke locatie en datum.

        Args:
            location: De locatie waarvoor maandata wordt opgevraagd
            datum: De datum waarvoor maandata wordt opgevraagd

        Returns:
            Een dictionary met maandata of foutinformatie
        """
        try:
            datum_str = datum.strftime("%Y%m%d")
            params = {
                'key': self.api_sleutel,
                'location': f"{location.longitude},{location.latitude}",
                'date': datum_str
                # 'lang': 'nl'  # Optioneel: stel taal in
            }

            response = self.sessie.get(self.api_host, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get('code') != '200':
                bericht = data.get('message', 'Onbekende fout')
                raise ValueError(f"API Response Fout {data.get('code')}: {bericht}")

            return self._verwerk_maan_data(data)

        except requests.RequestException as e:
            logger.error(f"Netwerkfout bij ophalen maandata: {e}")
            return {'error': f"Netwerkfout: {str(e)}"}
        except ValueError as e:
            logger.error(f"API fout bij ophalen maandata: {e}")
            return {'error': str(e)}
        except Exception as e:
            logger.error(f"Onverwachte fout bij ophalen maandata: {e}")
            return {'error': f"Onverwachte fout: {str(e)}"}

    def _verwerk_maan_data(self, data: Dict) -> Dict:
        """Verwerk de ruwe API-response naar een gestructureerd formaat."""
        return {
            'maanopkomst': data.get('moonrise', 'N/B'),
            'maanondergang': data.get('moonset', 'N/B'),
            'maanfasen': [
                {
                    'tijd': fase.get('fxTime', 'N/B'),
                    'waarde': fase.get('value', 'N/B'),
                    'naam': fase.get('name', 'N/B'),
                    'verlichting': fase.get('illumination', 'N/B'),
                    'icoon': fase.get('icon', 'N/B')
                }
                for fase in data.get('moonPhase', [])
            ]
        }

class GebedsTijdenCalculator:
    """Bereken gebedstijden en haal maandata op."""

    METHODEN = {
        'standaard': StandaardMethode(),
        'hanafi': HanafiMethode(),
    }

    def __init__(self, location: Location, methode: str = 'standaard'):
        self.location = location
        methode_lower = methode.lower()
        if methode_lower not in self.METHODEN:
            raise ValueError(f"Niet-ondersteunde methode. Kies uit: {', '.join(self.METHODEN.keys())}")
        self.methode = self.METHODEN[methode_lower]
        self.maan_provider = MaanDataProvider(API_KEY)

    def bereken_gebedstijden(self, datum: Union[date, datetime],
                             formaat: str = '24u') -> Dict[str, Union[str, Dict]]:
        """
        Bereken gebedstijden en haal maandata op voor een specifieke datum.

        Args:
            datum: De datum waarvoor gebedstijden worden berekend
            formaat: Tijdsformaat ('24u' of '12u')

        Returns:
            Een dictionary met gebedstijden, maandata en datuminformatie
        """
        try:
            if isinstance(datum, date) and not isinstance(datum, datetime):
                datum = datetime.combine(datum, datetime.min.time())

            # Zet datum naar UTC
            datum_utc = datum.replace(tzinfo=ZoneInfo('UTC'))

            # Bereken gebedstijden
            calculator = AstronomischeBerekeningen(self.location, datum_utc)
            tijden = calculator.calculate_prayer_times(self.methode)

            # Formatteer tijden
            geformatteerde_tijden = {
                gebed: self._formatteer_tijd(tijd, formaat)
                for gebed, tijd in tijden.items()
            }

            # Haal maan data op
            maan_data = self.maan_provider.haal_maan_data(self.location, datum_utc)

            return {
                **geformatteerde_tijden,
                'maan': maan_data,
                'gregoriaanse_datum': datum_utc.strftime("%Y-%m-%d"),
                'hijri_datum': self._krijg_hijri_datum(datum_utc)
            }

        except Exception as e:
            logger.error(f"Fout bij berekenen gebedstijden: {e}")
            raise

    @staticmethod
    def _formatteer_tijd(tijd: float, formaat: str = '24u') -> str:
        """Formatteer tijd naar gewenst formaat."""
        if math.isnan(tijd):
            return 'Ongeldig'

        uren = int(tijd)
        minuten = int(round((tijd - uren) * 60))

        if minuten == 60:
            uren = (uren + 1) % 24
            minuten = 0

        if formaat == '12u':
            periode = 'AM' if uren < 12 else 'PM'
            weergave_uren = uren % 12 or 12
            return f"{weergave_uren:02d}:{minuten:02d} {periode}"
        return f"{uren:02d}:{minuten:02d}"

    @staticmethod
    def _krijg_hijri_datum(datum: datetime) -> str:
        """Converteer Gregoriaanse datum naar Hijri datum."""
        try:
            gregoriaans = Gregorian(datum.year, datum.month, datum.day)
            hijri = gregoriaans.to_hijri()
            return f"{hijri.day} {hijri.month_name()} {hijri.year} AH"
        except Exception as e:
            logger.error(f"Fout bij conversie naar Hijri datum: {e}")
            return "Ongeldige Hijri datum"

def test_locaties() -> List[Location]:
    """Genereer een lijst met testlocaties."""
    return [
        Location(52.3676, 4.9041, "UTC", 2, "Amsterdam, Nederland"),
        Location(21.4225, 39.8262, "UTC", 277, "Mekka, Saoedi-Arabië"),
        Location(31.5497, 74.3436, "UTC", 217, "Lahore, Pakistan"),
        Location(-26.2041, 28.0473, "UTC", 1753, "Johannesburg, Zuid-Afrika"),
        Location(39.9042, 116.4074, "UTC", 43, "Beijing, China"),
        Location(55.7558, 37.6173, "UTC", 156, "Moscow, Rusland"),
        Location(5.8520, -55.2038, "UTC", 1, "Paramaribo, Suriname")
    ]

def main():
    """Hoofdfunctie voor het testen van de gebedstijden calculator."""
    test_datums = [
        datetime.now(ZoneInfo('UTC')),
        datetime.now(ZoneInfo('UTC')) + timedelta(days=1)
    ]

    locaties = test_locaties()
    methoden = list(GebedsTijdenCalculator.METHODEN.keys())

    for test_datum in test_datums:
        print(f"\nGebedstijden voor {test_datum.strftime('%Y-%m-%d')} GMT")
        print("=" * 50)

        for locatie in locaties:
            print(f"\nLocatie: {locatie.name} (Hoogte: {locatie.elevation}m)")
            print("-" * 50)

            for methode in methoden:
                try:
                    calculator = GebedsTijdenCalculator(locatie, methode)
                    tijden = calculator.bereken_gebedstijden(test_datum, formaat='24u')

                    print(f"\nJuridische Methode: {GebedsTijdenCalculator.METHODEN[methode].naam}")
                    print(f"Hijri Datum      : {tijden['hijri_datum']}")
                    print("-" * 30)

                    for gebed, tijd in tijden.items():
                        if gebed not in ['gregoriaanse_datum', 'hijri_datum', 'maan']:
                            print(f"{gebed.capitalize():10}: {tijd}")

                    # Print maan data
                    maan = tijden.get('maan', {})
                    if 'error' in maan:
                        print(f"\nMaan Data Fout: {maan['error']}")
                    else:
                        print("\nMaan Data:")
                        print(f"  Maanopkomst    : {maan.get('maanopkomst', 'N/B')}")
                        print(f"  Maanondergang  : {maan.get('maanondergang', 'N/B')}")
                        print("  Maanfasen:")
                        for fase in maan.get('maanfasen', []):
                            print(f"    Tijd         : {fase.get('tijd', 'N/B')}")
                            print(f"    Waarde       : {fase.get('waarde', 'N/B')}")
                            print(f"    Naam         : {fase.get('naam', 'N/B')}")
                            print(f"    Verlichting  : {fase.get('verlichting', 'N/B')}%")
                            print(f"    Icoon        : {fase.get('icoon', 'N/B')}")
                            print("    ---")

                except Exception as e:
                    logger.error(f"Fout voor {locatie.name}, methode {methode}: {e}")
                    print(f"\nKan gebedstijden niet berekenen voor methode {methode}.")

if __name__ == "__main__":
    main()
