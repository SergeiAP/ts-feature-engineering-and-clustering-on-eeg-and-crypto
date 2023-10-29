import pandas as pd
from typing import Callable


def download_ticket_price(func: Callable,
                          tickets: list,
                          period: int,
                          num_of_tickets: int,
                          to_ts: str) -> pd.DataFrame:
    """_summary_

    Args:
        func (Callable): _description_
        tickets (list): _description_
        period (int): _description_
        num_of_tickets (int): _description_
        to_ts (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    ticket_vals = {}
    counter = 0
    for ticket in tickets:
        ticket_info = func(ticket, currency='USD', limit=period, toTs=to_ts)
        if ticket_info is not None:
            ticket_info = pd.DataFrame(ticket_info)
            if counter == 0:
                ticket_vals['time'] = pd.to_datetime(ticket_info['time'], unit='s')
            # For simplicity will discover mean values of tickets 
            ticket_vals[ticket] = (ticket_info['high'] + ticket_info['low']) / 2
            counter += 1
            if counter == num_of_tickets:
                break
            
    ticket_price_day = pd.DataFrame(ticket_vals)
    ticket_price_day.set_index("time", inplace=True, drop=True)
    return ticket_price_day
