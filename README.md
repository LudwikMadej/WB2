# WB2 - CLIP concept debiasing

Projekt bada, jak usunąć niechciane biasy z modelu CLIP ViT-L/14 (model odpowiedniego rozmiaru - ma kilkanaście warstw na których można zrobić debiasing, ale wciąż działa w miarę szybko na słabszym sprzęcie). Punktem wyjścia są koncepty (np. okularów) na zbiorze CelebA.

## Struktura

```
checkpoints/   # eksperymenty pogrupowane per checkpoint - opis w README każdego z nich
software/      # współdzielony kod pomocniczy (ładowanie danych, TorchLR)
scripts/       # narzędzia
data/          # obrazy CelebA, metadata, aktywacje - poza gitem, pobierane przez rclone
```

## Instalacja

Zależności projektu są zdefiniowane w `pyproject.toml` - to standardowy plik konfiguracyjny Pythona (PEP 621), który zastępuje ręcznie pisane `requirements.txt`. Opisuje jakich pakietów projekt potrzebuje, w jakich wersjach i skąd je pobrać. Na jego podstawie generowany jest `uv.lock` (dokładne, zreprodukowane wersje wszystkich zależności) oraz `requirements.txt` (dla środowisk, które nie obsługują uv).

Do instalacji używamy `uv` - nowoczesnego zamiennika `pip`, napisanego w Ruście. Jest wielokrotnie szybszy od pip i sam zarządza wirtualnym środowiskiem, więc nie trzeba ręcznie robić `python -m venv`.

**Lokalnie i na RunPodzie (zalecane):**

```bash
pip install uv
uv sync --extra jupyter
.venv/bin/python -m ipykernel install --user --name wb2 --display-name "WB2"
```

Trzecia komenda rejestruje środowisko jako kernel Jupytera. Bez niej RunPod uruchomi notatnik na systemowym Pythonie i dostaniesz `ModuleNotFoundError`.

Po instalacji: **Kernel → Change Kernel → WB2** → Restart.

**Google Colab:**

```bash
pip install -r requirements.txt
```

## Dane

Katalog `data/` nie jest w gicie (jest za duży). Dane trzymamy w Cloudflare R2 i synchronizujemy przez `rclone`.

Instalacja rclone (jednorazowo):

```bash
# macOS
brew install rclone

# Linux
curl https://rclone.org/install.sh | sudo bash
```

```powershell
# Windows (PowerShell jako administrator)
winget install Rclone.Rclone
```

Pobieranie i wysyłanie danych:

```bash
./scripts/sync_data.sh           # pobierz
./scripts/sync_data.sh upload    # wyślij lokalne zmiany
```

Klucze API są wbudowane w skrypt - nie trzeba nic konfigurować.

## Aktualizacja zależności

Jeśli zmienisz `pyproject.toml`, wygeneruj nowy `requirements.txt`:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project \
  | (echo '--extra-index-url https://download.pytorch.org/whl/cu126'; cat) \
  > requirements.txt
```

Nie edytuj `requirements.txt` ręcznie.
