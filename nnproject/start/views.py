from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import SavingForm
from django.urls import reverse
from math import pow
import locale
locale.setlocale(locale.LC_ALL, '')

def index(request):
    error = ''
    # if request.method == 'POST':
    #     target = request.POST.get('target')
    #     period = request.POST.get('period')
    #     inflation = request.POST.get('inflation')
    #     profit = request.POST.get('profit')
    #     start_sum = request.POST.get('start_sum')
    #     return HttpResponse("Твоя цел")
    # else:
    #     savingForm = SavingForm()
    #     return render(request, 'start/index.html', {'form': savingForm})
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = SavingForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # form.save()
            target = form.cleaned_data['target']
            period = form.cleaned_data['period']
            inflation = form.cleaned_data['inflation']
            profit = form.cleaned_data['profit']
            start_sum = form.cleaned_data['start_sum']
            return redirect(reverse(
                'result') + f"?target={target}&period={period}&inflation={inflation}&profit={profit}&start_sum={start_sum}")
        else:
            error = 'ERROR'

    savingForm = SavingForm()
    data = {
        'title': 'Калькулятор вашей цели',
        'form': savingForm,
        'error': error
    }
    return render(request, 'start/index.html', data)


def result(request):
    target = int(request.GET.get('target'))
    period = int(request.GET.get('period'))
    inflation = float(request.GET.get('inflation'))
    profit = float(request.GET.get('profit')) / 100
    start_sum = int(request.GET.get('start_sum'))
    profit_in_period = pow(1 + profit, period)
    global future_value, present_value, r_without_start_sum_year, r_without_start_sum_month, \
        r_with_start_sum_year, r_with_start_sum_month

    future_value = int(target * pow(1 + inflation / 100, period))  # цена в будущем с учетом инфляции
    present_value = int(future_value / profit_in_period)  # сумма,к-ую необ внести сейчас (единоразово)
    data = {
            'target': locale.format('%d', target, grouping=True),
            'period': period,
            'inflation': inflation,
            'profit': profit * 100,
            'start_sum': start_sum,
            'future_value': locale.format('%d',future_value,grouping=True),
            'present_value': locale.format('%d',present_value, grouping=True),
    }

    if (start_sum == 0):  # без учета стартового капитала
        r_without_start_sum_year = int(future_value / (
                (profit_in_period - 1) / profit))  # периодические внесения на счет с учетом доходности В ГОД
        r_without_start_sum_month = int(r_without_start_sum_year / 12)
        data['r_without_start_sum_year'] = locale.format('%d', r_without_start_sum_year,  grouping=True),
        data['r_without_start_sum_month'] = locale.format('%d', r_without_start_sum_month,  grouping=True),
    else:
        profit_start_sum = start_sum * profit_in_period  # сколько можно заработать только со стартового капитола
        other_sum = future_value - profit_start_sum
        r_with_start_sum_year = int(other_sum / (
                (profit_in_period - 1) / profit))  # периодические внесения на счет с учетом доходности В ГОД
        r_with_start_sum_month = int(r_with_start_sum_year / 12)
        data['r_with_start_sum_year'] = locale.format('%d', r_with_start_sum_year,  grouping=True)
        data['r_with_start_sum_month'] = locale.format('%d', r_with_start_sum_month,  grouping=True)

    return render(request, 'start/result.html', data)
