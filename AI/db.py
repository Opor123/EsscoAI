from sqlalchemy import create_engine,Column,Integer,Text,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker
from datetime import datetime
from pathlib import Path
Base=declarative_base()

class QAPair(Base):
    __tablename__ = 'QAPair'
    id=Column(Integer,primary_key=True,autoincrement=True)
    question=Column(Text,nullable=False)
    answer=Column(Text,nullable=False)

class Feedback(Base):
    __tablename__ = 'Feedback'
    id=Column(Integer,primary_key=True,autoincrement=True)
    question=Column(Text,nullable=False)
    model_answer=Column(Text,nullable=False)
    correct_answer=Column(Text,nullable=False)
    label=Column(Text,nullable=True)
    timeStamp=Column(DateTime,default=datetime.utcnow)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR=PROJECT_ROOT / 'Data'
DATA_DIR.mkdir(parents=True,exist_ok=True)
DB_PATH=DATA_DIR / 'essco_ai.db'

engine=create_engine(f'sqlite:///{DB_PATH}')

Session=sessionmaker(bind=engine)

Base.metadata.create_all(engine)