echo "Setup project..."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python "$SCRIPT_DIR"/manage.py collectstatic
python "$SCRIPT_DIR"/manage.py migrate
python "$SCRIPT_DIR"/manage.py createsuperuser
