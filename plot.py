import math

weights_in = [float(x.strip()) for x in open('weights.txt')]
max_abs = max(abs(x) for x in weights_in)
weights = [0 for x in range(640)]

window_size = 100
input_losses = [float(x.strip()) for x in open('losses.txt') if not x.startswith('--')]
losses = [sum(input_losses[i:i+window_size])/window_size
    for i in range(len(input_losses) - window_size)]

for i in range(len(weights)):
    board, r = divmod(i, 80)
    row, col = divmod(r, 10)
    grp, board = divmod(board, 4)
    j = grp * 320 + (7 - row) * 40 + (9 - col) * 4 + board
    weights[i] = weights_in[j]

table_names = [
    'white pyramid',
    'white anubis',
    'white scarab',
    'white pharaoh',
    'red pyramid',
    'red anubis',
    'red scarab',
    'red pharaoh'
]

print('''
    <!DOCTYPE html>
    <html>
        <head>
        <style>
            body {
                background-color: #000;
                color: #fff;
                font-family: Iosevka, monospace;
            }
        </style>
        </head>
        <body>
''')

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def gradient(x):
    if x <= 0:
        return gradient_negative(-x)
    else:
        return gradient_positive(x)

# khet colors
#def gradient_negative(x):
#    return 0.1 + 0.8 * x, 0.1 - 0.1 * x, 0.1 - 0.1 * x
#def gradient_positive(x):
#    return 0.1 + 0.7 * x, 0.1 + 0.8 * x, 0.1 + 0.9 * x

# normal colors
def gradient_negative(x):
    return 0.1 + 0.8 * x, 0.1 + 0.1 * x, 0.1 - 0.1 * x
def gradient_positive(x):
    return 0.1, 0.1 + 0.9 * x, 0.1 + 0.4 * x

def put_board(board):
    print('<svg viewbox="0 0 200 160" width="200" height="160">')
    for row in range(8):
        for col in range(10):
            i = board * 80 + row * 10 + col
            w = weights[i]
            color = w / max_abs
            r, g, b = gradient(color)
            print('''
                <rect x="{x}" y="{y}" rx="1" ry="1" width="20" height="20"
                    fill="rgb({r}% {g}% {b}%)"/>
            '''.format(
                x = col * 20,
                y = row * 20,
                r = 100 * r,
                g = 100 * g,
                b = 100 * b,
            ))
    print('</svg>')
    print('<p style="margin: 0 0 30px 0">{}. {}</p>'.format(board, table_names[board]))

print('<table style="margin: 80px">')
print('<tbody>')
for row in range(2):
    print('<tr>')
    for col in range(4):
        print('<td>')
        put_board(row * 4 + col)
        print('</td>')
    print('</tr>')
print('</tbody>')
print('</table>')

print('<div style="margin: 80px">')
min_loss = min(min(losses), 0.8)
max_loss = max(max(losses), 1.0)
print('<svg viewbox="0 0 800 400" width="800" height="400">')
def plot(i, loss):
    x = 5 + 790 * i / len(losses)
    y = 5 + 390 * (1 - (loss - min_loss) / (max_loss - min_loss))
    return x, y
for i in range(40):
    loss = i / 20.
    print('''<path
        fill="non"
        stroke="rgb({color}%, {color}%, {color}%)"
        stroke-width="1"
        stroke-linecap="round"
        stroke-linejoin="round"
        {dashes}
        d="M 0 {y} L 800 {y}"
        />'''.format(
            y = plot(0, loss)[1],
            color = 40 if i % 20 != 0 else 70,
            dashes = 'stroke-dasharray="2 2"' if i % 20 != 0 else ''
        ))
print('''<path
    fill="none"
    stroke="rgb(100%, 100%, 100%)"
    stroke-width="1"
    stroke-linecap="round"
    stroke-linejoin="round"
    d="
    ''')
for i, loss in enumerate(losses):
    x, y = plot(i, loss)
    print('{} {} {} '.format('M' if i == 0 else 'L', x, y))
print('"/>')
print('</svg>')
print('</div>')

print('''
</body>
</html>
''')