from django.db import models


class Band(models.Model):
  

    name = models.CharField(max_length=200)
    can_rock = models.BooleanField(default=True)


class Member(models.Model):
  

    name = models.CharField("Member's name", max_length=200)
    instrument = models.CharField(
        choices=(
            ("g", "Guitar"),
            ("b", "Bass"),
            ("d", "Drums"),
        ),
        max_length=1,
    )
    band = models.ForeignKey("Band")