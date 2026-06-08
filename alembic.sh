#!/bin/bash
# A quick reference script for running Alembic database migrations.

# 1. Autogenerate a new migration script based on model changes
# Usage: ./alembic.sh make "description of changes"
if [ "$1" == "make" ]; then
    alembic revision --autogenerate -m "$2"
    echo "Migration script generated in alembic/versions/"
fi

# 2. Upgrade the database to the latest version (head)
# Usage: ./alembic.sh up
if [ "$1" == "up" ]; then
    alembic upgrade head
    echo "Database upgraded to latest version."
fi

# 3. Downgrade/rollback the database by one version
# Usage: ./alembic.sh down
if [ "$1" == "down" ]; then
    alembic downgrade -1
    echo "Database downgraded by 1 revision."
fi
