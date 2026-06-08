from sqlalchemy.orm import Session
from app import models
from app.core.security import get_password_hash

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, username: str, password: str):
    hashed_password = get_password_hash(password)
    db_user = models.User(username=username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def seed_default_user(db: Session):
    user = get_user_by_username(db, "admin")
    if not user:
        create_user(db, "admin", "password#123")
