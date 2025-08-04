import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import torch.nn.functional as F  # ganz oben importieren!

# Segmentdaten für Zahlen 0-9
segment_data = {
    0: [1,1,1,1,1,1,0],
    1: [0,1,1,0,0,0,0],
    2: [1,1,0,1,1,0,1],
    3: [1,1,1,1,0,0,1],
    4: [0,1,1,0,0,1,1],
    5: [1,0,1,1,0,1,1],
    6: [1,0,1,1,1,1,1],
    7: [1,1,1,0,0,0,0],
    8: [1,1,1,1,1,1,1],
    9: [1,1,1,1,0,1,1],
}

X = torch.tensor(list(segment_data.values()), dtype=torch.float32)
y = torch.tensor(list(segment_data.keys()), dtype=torch.long)

class SevenSegmentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SevenSegmentApp:
    def __init__(self, root):
        self.model = SevenSegmentNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.current_epoch = 0  # aktuelle Epoche speichern

        self.root = root
        self.root.title("Siebensegment KI")

        self.segment_buttons = []
        self.segment_state = [0]*7

        tk.Label(root, text="Klicke die Segmente:").pack()

        frame = tk.Frame(root)
        frame.pack()

        for i in range(7):
            var = tk.IntVar()
            btn = tk.Checkbutton(frame, text=f"S{i+1}", variable=var, indicatoron=False,
                                 command=self.update_segment_display)
            btn.var = var
            btn.grid(row=i//4, column=i%4)
            self.segment_buttons.append(btn)

        # 7-Segment-Anzeige (Canvas)
        self.canvas_width = 120
        self.canvas_height = 200
        self.segment_canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.segment_canvas.pack(pady=10)

        self.predict_label = tk.Label(root, text="Zahl erkannt: -")
        self.predict_label.pack()

        self.probs_label = tk.Label(root, text="Wahrscheinlichkeiten: -")
        self.probs_label.pack()

        self.train_button = tk.Button(root, text="Netz trainieren", command=self.train_model)
        self.train_button.pack()

        self.predict_button = tk.Button(root, text="Zahl vorhersagen", command=self.predict)
        self.predict_button.pack()

        tk.Label(root, text="Startneuron (1-7, optional):").pack()
        self.input_neuron_entry = tk.Entry(root)
        self.input_neuron_entry.pack()
        self.weight_change_button = tk.Button(root, text="Gewichtsänderung anzeigen", command=self.show_weight_change)
        self.weight_change_button.pack(pady=5)

        self.loss_vals = []
        self.probs_vals = [0]*10  # Aktuelle Wahrscheinlichkeiten für 10 Klassen
        
        self.fig, (self.ax_loss, self.ax_probs) = plt.subplots(2, 1, figsize=(4, 4))
        self.fig.subplots_adjust(hspace=1)  # hier Abstand einstellen
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.fig2, self.ax2 = plt.subplots(figsize=(4, 2))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.root)
        self.canvas2.get_tk_widget().pack(pady=(10,20))  # Etwas Abstand zum ersten Diagramm

        # 7 Segmente initial zeichnen
        self.segment_shapes = self.draw_7segment()
        self.update_segment_display()

    def update_state(self):
        for i, btn in enumerate(self.segment_buttons):
            self.segment_state[i] = btn.var.get()

    def predict(self):
        self.update_state()
        input_tensor = torch.tensor(self.segment_state, dtype=torch.float32)
        with torch.no_grad():
            h1 = torch.relu(self.model.fc1(input_tensor))
            out = self.model.fc2(h1)
            probs = torch.softmax(out, dim=0)  # Softmax normalisiert Output zu Wahrscheinlichkeiten
            prediction = torch.argmax(probs).item()
        
        self.predict_label.config(text=f"Zahl erkannt: {prediction}")
        probs_str = ", ".join(f"{i}:{p:.3f}" for i, p in enumerate(probs))
        self.probs_label.config(text=f"Wahrscheinlichkeiten: {probs_str}")

        self.probs_vals = probs.tolist()  # Wahrscheinlichkeiten speichern
        self.update_plot()  # Aktualisiere Plot mit neuen Wahrscheinlichkeiten

        self.show_network_graph(input_tensor, h1, out, self.current_epoch, self.segment_state)

    def train_model(self):
        #self.loss_vals.clear()
        for epoch in range(200):
            output = self.model(X)
            loss = self.criterion(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_vals.append(loss.item())
            self.current_epoch += 1

            if epoch % 10 == 0:
                self.update_plot()

    def update_plot(self):
        # Loss-Plot aktualisieren
        self.ax_loss.clear()
        self.ax_loss.plot(self.loss_vals, label="Loss")
        self.ax_loss.set_title("Lernverlauf")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()

        # Wahrscheinlichkeiten als Balkendiagramm darstellen
        self.ax_probs.clear()
        self.ax_probs.bar(range(10), self.probs_vals, color='skyblue')
        self.ax_probs.set_title("Wahrscheinlichkeiten der Klassen")
        self.ax_probs.set_xlabel("Zahl")
        self.ax_probs.set_ylabel("Wahrscheinlichkeit")
        self.ax_probs.set_ylim(0, 1)

        self.canvas.draw()

    # Letzte 200 Epochen
        self.ax2.clear()
        last_200 = self.loss_vals[-200:] if len(self.loss_vals) > 200 else self.loss_vals
        start_epoch = self.current_epoch - len(last_200)
        self.ax2.plot(range(start_epoch, self.current_epoch), last_200, label="Loss letzte 200 Epochen", color='orange')
        self.ax2.set_title("Lernverlauf (letzte 200 Epochen)")
        self.ax2.set_xlabel("Epoche")
        self.ax2.set_ylabel("Loss")
        self.ax2.legend()
        self.canvas2.draw()
    

    def show_network_graph(self, input_tensor, h1, out, epoch=0, segments=None):
        win = tk.Toplevel(self.root)

        width = 800
        height = 600

        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()

        # Position: rechts oben, 50 Pixel Abstand vom Rand
        x = screen_width - width - 50
        y = 50

        win.geometry(f"{width}x{height}+{x}+{y}")

        seg_str = ",".join(str(s) for s in segments) if segments is not None else ""
        win.title(f"Neuronale Aktivierung - Epoche({epoch}) Segmente({seg_str})")

        fig, ax = plt.subplots(figsize=(10, 6))
        layers = [7, 16, 10]
        positions = []

        for i, n in enumerate(layers):
            x = i * 3
            ys = [(j + 1) * (6 / (n + 1)) for j in range(n)]
            positions.append([(x, y) for y in ys])

        weights1 = self.model.fc1.weight.detach().numpy()
        weights2 = self.model.fc2.weight.detach().numpy()

        for i, (x0, y0) in enumerate(positions[0]):
            for j, (x1, y1) in enumerate(positions[1]):
                weight = weights1[j][i]
                color = 'green' if weight > 0 else 'red'
                width = max(0.1, abs(weight) * 1.5)
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=width)

        for i, (x0, y0) in enumerate(positions[1]):
            for j, (x1, y1) in enumerate(positions[2]):
                weight = weights2[j][i]
                color = 'green' if weight > 0 else 'red'
                width = max(0.1, abs(weight) * 1.5)
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=width)

        # NEU: Softmax auf Output anwenden für Visualisierung
        probs = F.softmax(out, dim=0)

        for l, layer_pos in enumerate(positions):
            for i, (x, y) in enumerate(layer_pos):
                act = None
                if l == 0:
                    act = input_tensor[i].item()
                elif l == 1:
                    act = h1[i].item()
                elif l == 2:
                    act = probs[i].item()  # <- GEÄNDERT: jetzt normalisierte Werte

                circ = patches.Circle(
                    (x, y), 0.2,
                    color='skyblue' if act is None else 'orange' if act > 0.5 else 'lightgrey'
                )
                ax.add_patch(circ)
                ax.text(x, y, f"{act:.2f}" if act is not None else "", ha='center', va='center', fontsize=8)

        ax.set_xlim(-1, 7)
        ax.set_ylim(0, 6.5)
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack()

        # --- MSE-Berechnung und Anzeige ---
        predicted_class = torch.argmax(probs).item()
        target_one_hot = torch.zeros_like(probs)
        target_one_hot[predicted_class] = 1.0
        mse = torch.mean((probs - target_one_hot) ** 2).item()

        mse_text = f"MSE: {mse:.6f}\n" 
        mse_label = tk.Label(win, text=mse_text, font=("Arial", 12), fg="blue", pady=10)
        mse_label.pack()

        canvas.draw()
    
    def show_weight_change(self):
        try:
            index_str = self.input_neuron_entry.get().strip()
            if index_str:
                input_index = int(index_str) - 1  # Eingabe 1-basiert, Index 0-basiert
                if not 0 <= input_index < 7:
                    raise ValueError
            else:
                input_index = None  # Kein Filter → alle
        except ValueError:
            tk.messagebox.showerror("Fehler", "Bitte gib eine Zahl zwischen 1 und 7 ein.")
            return

        # Vorher-Zustand speichern
        old_weights = self.model.fc1.weight.clone().detach()
        
        # Einmal trainieren
        output = self.model(X)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Nachher-Zustand
        new_weights = self.model.fc1.weight.clone().detach()

        # Differenz
        delta = new_weights - old_weights

        # --- Fenster erstellen ---
        win = tk.Toplevel(self.root)
        title = f"Gewichtsveränderung für Input-Neuron {input_index+1}" if input_index is not None else "Gewichtsveränderung (alle)"
        win.title(title)

        if input_index is not None:
            fig_width = 10
            fig_height = 5
        else:
            fig_width = max(14, delta.numel() / 8)  # größer bei vielen Balken
            fig_height = 6

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if input_index is not None:
            # Nur die Spalte für das gewählte Input-Neuron
            changes = delta[:, input_index].numpy()
            labels = [f"I{input_index+1}→H{j+1}" for j in range(len(changes))]
        else:
            # Alle Verbindungen (flach)
            changes = delta.T.flatten().numpy()
            labels = [f"I{i+1}→H{j+1}" for i in range(delta.size(1)) for j in range(delta.size(0))]


        ax.bar(range(len(changes)), changes, color=["green" if v >= 0 else "red" for v in changes])
        ax.set_title("Veränderung der Gewichte nach 1 Epoche")
        ax.set_xlabel("Verbindung")
        ax.set_ylabel("Δ Gewicht")
        ax.set_xticks(range(len(changes)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack()
        canvas.draw()




    def draw_7segment(self):
        # 7 Segmente als Canvas-Objekte, Rückgabe Liste der IDs
        # Positionen und Größen orientiert an klassischer 7-Segment-Anzeige
        s = 20  # Segment-Länge
        t = 5   # Segment-Dicke
        cx = self.canvas_width//2
        cy = self.canvas_height//2

        segments = []

        # Segment A - oben horizontal
        segA = self.segment_canvas.create_rectangle(cx - s, cy - 3*s, cx + s, cy - 3*s + t, fill='red')
        segments.append(segA)

        # Segment B - oben rechts vertikal
        segB = self.segment_canvas.create_rectangle(cx + s, cy - 3*s + t, cx + s + t, cy - s, fill='red')
        segments.append(segB)

        # Segment C - unten rechts vertikal
        segC = self.segment_canvas.create_rectangle(cx + s, cy - s + t, cx + s + t, cy + s, fill='red')
        segments.append(segC)

        # Segment D - unten horizontal
        segD = self.segment_canvas.create_rectangle(cx - s, cy + s, cx + s, cy + s + t, fill='red')
        segments.append(segD)

        # Segment E - unten links vertikal
        segE = self.segment_canvas.create_rectangle(cx - s - t, cy - s + t, cx - s, cy + s, fill='red')
        segments.append(segE)

        # Segment F - oben links vertikal
        segF = self.segment_canvas.create_rectangle(cx - s - t, cy - 3*s + t, cx - s, cy - s, fill='red')
        segments.append(segF)

        # Segment G - Mitte horizontal
        segG = self.segment_canvas.create_rectangle(cx - s, cy - s, cx + s, cy - s + t, fill='red')
        segments.append(segG)

        return segments

    def update_segment_display(self):
        self.update_state()
        # Segment ON = grün, OFF = rot
        for i, seg_id in enumerate(self.segment_shapes):
            color = 'lime' if self.segment_state[i] else 'darkred'
            self.segment_canvas.itemconfig(seg_id, fill=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SevenSegmentApp(root)
    root.mainloop()
