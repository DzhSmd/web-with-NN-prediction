from django import forms

class SavingForm(forms.Form):
    target = forms.IntegerField(label="Сумма цели",min_value=1, initial=100000)#,widget=forms.NumberInput(attrs={"class":"myfield"}))
    period = forms.IntegerField(label="Количество лет",min_value=1, max_value= 100,initial=10)#, min_value=1)#, widget=forms.NumberInput(attrs={'class': 'my-custom-class'}))
    inflation = forms.FloatField(label="Среднегодовая ставка инфляции, %",min_value=0, max_value= 1000,initial=8)
    profit = forms.FloatField(label="Среднегодовая ставка доходности, %",min_value=1, max_value= 1000,initial=15)
    start_sum = forms.IntegerField(label="Стартовый капитал",min_value=0, initial=10000)

# отобразить числа по разрядам
