# ==============================================================================
# Génération de dataset d'arbres phylogénétiques avec annotations COCO
# ==============================================================================
# Ce script génère un dataset d'arbres phylogénétiques au format PNG avec des
# annotations de points (corners, nodes, leaves) dans une structure COCO JSON.
#
# Structure COCO générée:
#   - Catégorie 1 = leaf (feuille)
#   - Catégorie 2 = internal_node (nœud interne)
#   - Catégorie 3 = corner (coin/virage)
#
# Sorties:
#   - Images PNG des arbres phylogénétiques
#   - Fichier annotations.json au format COCO
# ==============================================================================

library(ape)
library(jsonlite)

# ==============================================================================
# PARAMÈTRES DE CONFIGURATION
# ==============================================================================

# --- Chemins et répertoires ---
out_dir <- "/home/cactus/Desktop/PFE/12-dataset-r-clean/out_dataset"
img_dir <- file.path(out_dir, "images")
dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)

# --- Dataset ---
N <- 10                # Nombre d'arbres à générer
set.seed(1)            # Graine aléatoire pour reproductibilité

# --- Dimensions des images ---
W <- 1000              # Largeur en pixels
H <- 1000              # Hauteur en pixels

# --- Annotations COCO ---
radius <- 6            # Rayon des points annotés (en pixels)
k_poly <- 16           # Nombre de points pour approximer un cercle (polygone)

# --- Labels des feuilles ---
P_LABELS <- 0          # Probabilité qu'un arbre ait des labels visibles (0-1)
P_TIP_LABELED <- 1     # Si labels visibles: proportion de feuilles labellisées (0-1)
LABEL_LEN <- 6         # Longueur des noms de labels (caractères aléatoires)

# --- Arbres non binaires (polytomies) ---
P_POLYTOMY <- 0.3      # Probabilité de créer des polytomies (0-1)
TOL_MIN <- 0.01        # Tolérance minimale pour regrouper les branches
TOL_MAX <- 0.50        # Tolérance maximale pour regrouper les branches

# --- Espacement vertical ---
TIP_SPACING_MIN <- 0.5 # Espacement minimal entre feuilles (<1 = serré)
TIP_SPACING_MAX <- 0.5 # Espacement maximal entre feuilles (>1 = espacé)

# --- Racine de l'arbre --- (marche pas)
P_ROOT_EDGE <- 1.0     # Probabilité d'ajouter une branche racine (0-1)
ROOT_EDGE_MIN <- 0.2   # Longueur minimale de la branche racine
ROOT_EDGE_MAX <- 0.5   # Longueur maximale de la branche racine

# --- Style visuel ---
EDGE_W_MIN <- 1        # Épaisseur minimale des branches
EDGE_W_MAX <- 10       # Épaisseur maximale des branches
P_USE_EDGE_LEN <- 0    # Probabilité d'utiliser les longueurs de branches réelles (0-1)
DIRECTIONS <- c("rightwards")  # Direction de l'arbre (rightwards, leftwards, upwards, downwards)

# --- Position des labels ---
P_LABEL_OFFSET_ZERO <- 0.5  # Probabilité que les labels soient collés aux feuilles (0-1)
LABEL_OFFSET_MIN <- 0.0     # Offset minimal des labels
LABEL_OFFSET_MAX <- 0.2     # Offset maximal des labels

# --- Marqueurs sur les nœuds --- (non utilisée actuellement remplacée par augment_dataset.py ou add_noise_dataset.py)
P_NODE_MARK <- 0       # Probabilité d'afficher un marqueur sur les nœuds (0-1)

# --- Règle graduée (non utilisée actuellement) ---
P_RULER_TOP <- 0       # Probabilité d'afficher une règle en haut
P_RULER_BOTTOM <- 0    # Probabilité d'afficher une règle en bas
RULER_BAND_FRAC <- 0.1      # Hauteur de la bande de règle (fraction)
RULER_XPAD_FRAC <- 0.06     # Marge gauche/droite de la règle (fraction)
RULER_TICK_FRAC <- 0.1      # Longueur des graduations (fraction)
RULER_LWD_MIN <- 2          # Épaisseur minimale de la règle
RULER_LWD_MAX <- 4          # Épaisseur maximale de la règle
RULER_MAX_MIN <- 0.6        # Valeur maximale minimale de la règle
RULER_MAX_MAX <- 4.0        # Valeur maximale maximale de la règle
RULER_STEP_CHOICES <- c(0.1, 0.2, 0.25, 0.5, 1.0)  # Pas possibles pour les graduations

draw_ruler <- function(mode) {
  #Dessine une règle graduée en haut ou en bas de l'arbre.
  if (mode == "none") return()
  
  usr <- par("usr")
  xmin <- usr[1]; xmax <- usr[2]
  ymin <- usr[3]; ymax <- usr[4]
  xspan <- xmax - xmin
  yspan <- ymax - ymin
  
  band <- yspan * RULER_BAND_FRAC
  tick <- band * RULER_TICK_FRAC
  
  y0 <- if (mode == "bottom") (ymin - band * 0.55) else (ymax + band * 0.55)
  
  x0 <- xmin + xspan * RULER_XPAD_FRAC
  x1 <- xmax - xspan * RULER_XPAD_FRAC
  
  lwd <- runif(1, RULER_LWD_MIN, RULER_LWD_MAX)
  
  maxv <- runif(1, RULER_MAX_MIN, RULER_MAX_MAX)
  step <- sample(RULER_STEP_CHOICES, 1)
  maxv <- ceiling(maxv / step) * step
  labels <- seq(maxv, 0, by = -step)
  n <- length(labels)
  
  old <- par(xpd = NA)
  
  segments(x0, y0, x1, y0, lwd = lwd, col = "black")
  
  for (i in 1:n) {
    t <- (i - 1) / (n - 1)
    xi <- x0 + (x1 - x0) * t
    
    segments(xi, y0, xi, y0 - tick, lwd = lwd, col = "black")
    
    lab <- format(labels[i], trim = TRUE, scientific = FALSE)
    text(xi, y0 - tick * 2.0, labels = lab, cex = 1.0)
  }
  
  par(old)
}


rand_label <- function(n = 6) {
  #Génère un label aléatoire de n lettres.
  paste0(sample(letters, n, replace = TRUE), collapse = "")
}


to_px <- function(x, y, usr, W, H, plt) {
  #Convertit les coordonnées plot en coordonnées pixel.
  x0 <- plt[1] * W
  x1 <- plt[2] * W
  y0 <- plt[3] * H
  y1 <- plt[4] * H
  
  px <- x0 + (x - usr[1]) / (usr[2] - usr[1]) * (x1 - x0)
  py <- y1 - (y - usr[3]) / (usr[4] - usr[3]) * (y1 - y0)
  
  list(px = px, py = py)
}


add_point_ann <- function(img_id, cat_id, cx, cy, r) {
  # Crée une annotation COCO pour un point circulaire.
  theta <- seq(0, 2*pi, length.out = k_poly)
  poly <- as.numeric(rbind(cx + r*cos(theta), cy + r*sin(theta)))
  bbox <- c(cx - r, cy - r, 2*r, 2*r)
  
  list(
    id = NULL, 
    image_id = img_id, 
    category_id = cat_id,
    segmentation = list(poly),
    bbox = bbox,
    area = pi * r * r,
    iscrowd = 0
  )
}

coco <- list(
  images = list(),
  annotations = list(),
  categories = list(
    list(id = 1, name = "leaf"),
    list(id = 2, name = "internal_node"),
    list(id = 3, name = "corner")
  )
)

ann_id <- 1   # Compteur d'annotations
img_id <- 1   # Compteur d'images

# ==============================================================================
# BOUCLE PRINCIPALE - GÉNÉRATION DES ARBRES
# ==============================================================================
pb <- txtProgressBar(min = 0, max = N, style = 3)
UPDATE_EVERY <- 25

for (i in 1:N) {
  
  nTips <- sample(10:80, 1) # Nombre de feuilles aléatoire entre 10 et 80
  tr <- rtree(nTips)
  
  nTip <- length(tr$tip.label)
  tr$tip.label <- replicate(nTip, rand_label(LABEL_LEN))
  
  # Décision d'afficher les labels
  show_labels <- (runif(1) < P_LABELS)
  if (show_labels && P_TIP_LABELED < 1.0) {
    keep <- runif(nTip) < P_TIP_LABELED
    tr$tip.label[!keep] <- ""
  }
  
  if (runif(1) < P_POLYTOMY) {
    tr <- multi2di(tr, random = TRUE)
    tr <- di2multi(tr, tol = runif(1, TOL_MIN, TOL_MAX))
  }
  
  use_edge_len <- (runif(1) < P_USE_EDGE_LEN)
  edge_w <- runif(1, EDGE_W_MIN, EDGE_W_MAX)
  direction <- sample(DIRECTIONS, 1)
  
  tip_spacing <- runif(1, TIP_SPACING_MIN, TIP_SPACING_MAX)
  ylim <- c(0.5, (nTip + 0.5) * tip_spacing)
  
  use_root_edge <- (runif(1) < P_ROOT_EDGE)
  root_edge_val <- if (use_root_edge) runif(1, ROOT_EDGE_MIN, ROOT_EDGE_MAX) else NULL
  
  fname <- sprintf("tree_%06d.png", img_id)
  fpath <- file.path(img_dir, fname)
  
  png(fpath, width = W, height = H, bg = "white")
  par(mar = c(2,2,2,2))
  
  if (use_root_edge) {
    tr$root.edge <- root_edge_val
  }
  
  # Dessin de l'arbre
  par(mar = c(6, 2, 6, 2)) 
  plot.phylo(
    tr,
    type = "phylogram",
    direction = direction,
    show.tip.label = FALSE,
    no.margin = FALSE,
    use.edge.length = use_edge_len,
    edge.width = edge_w,
    root.edge = use_root_edge,
  )
  
  ruler_mode <- "none"
  u <- runif(1)
  if (u < P_RULER_BOTTOM) {
    ruler_mode <- "bottom"
  } else if (u < P_RULER_BOTTOM + P_RULER_TOP) {
    ruler_mode <- "top"
  }
  draw_ruler(ruler_mode)
  
  if (show_labels) {
    
    lp <- get("last_plot.phylo", envir = .PlotPhyloEnv)
    
    tip_idx <- 1:nTip
    xx <- lp$xx[tip_idx]
    yy <- lp$yy[tip_idx]
    
    use_zero <- runif(nTip) < P_LABEL_OFFSET_ZERO
    
    offsets <- ifelse(
      use_zero,
      0,
      runif(nTip, LABEL_OFFSET_MIN, LABEL_OFFSET_MAX)
    )
    
    for (i in tip_idx) {
      if (tr$tip.label[i] == "") next
      
      text(
        x = xx[i] + offsets[i],
        y = yy[i],
        labels = tr$tip.label[i],
        adj = c(0, 0.5),
        cex = 0.8
      )
    }
  }
  
  lp <- get("last_plot.phylo", envir = .PlotPhyloEnv)
  
  nTip <- length(tr$tip.label)
  nodes <- (nTip + 1):(nTip + tr$Nnode)
  
  mask <- runif(length(nodes)) < P_NODE_MARK
  
  if (any(mask)) {
    points(
      lp$xx[nodes[mask]],
      lp$yy[nodes[mask]],
      pch = sample(c(21, 22, 24), sum(mask), replace = TRUE),
      bg  = "white",
      col = "black",
      cex = runif(sum(mask), 0.6, 1.4)
    )
  }
  
  xx <- lp$xx
  yy <- lp$yy
  usr <- par("usr")
  plt <- par("plt")
  
  dev.off()
  
  
  # CRÉATION DES ANNOTATIONS COCO
  
  nTip <- length(tr$tip.label)
  nNode <- tr$Nnode
  tips <- 1:nTip
  nodes <- (nTip + 1):(nTip + nNode)
  
  parents <- tr$edge[, 1]
  children <- tr$edge[, 2]
  
  nTip <- length(tr$tip.label)
  internal_parents <- unique(parents[parents > nTip])
  
  # Extraction des coins (corners)
  corner_list <- list()
  idx <- 1
  
  for (p in internal_parents) {
    ch <- children[parents == p]
    if (length(ch) == 0) next
    
    y_min <- min(yy[ch])
    y_max <- max(yy[ch])
    x_p <- xx[p]
    
    corner_list[[idx]] <- c(x_p, y_min); idx <- idx + 1
    corner_list[[idx]] <- c(x_p, y_max); idx <- idx + 1
  }
  
  corners <- do.call(rbind, corner_list)
  corners <- unique(corners)
  
  # Conversion des coordonnées en pixels
  tips_px <- to_px(xx[tips], yy[tips], usr, W, H, plt)
  nodes_px <- to_px(xx[nodes], yy[nodes], usr, W, H, plt)
  corners_px <- to_px(corners[,1], corners[,2], usr, W, H, plt)
  
  # Ajout de l'image au dataset COCO
  coco$images[[length(coco$images) + 1]] <- list(
    id = img_id, file_name = fname, width = W, height = H
  )
  
  # Ajout des annotations pour les feuilles (catégorie 1)
  for (j in seq_along(tips)) {
    ann <- add_point_ann(img_id, 1, tips_px$px[j], tips_px$py[j], radius)
    ann$id <- ann_id; ann_id <- ann_id + 1
    coco$annotations[[length(coco$annotations) + 1]] <- ann
  }
  
  # Ajout des annotations pour les nœuds internes (catégorie 2)
  for (j in seq_along(nodes)) {
    ann <- add_point_ann(img_id, 2, nodes_px$px[j], nodes_px$py[j], radius)
    ann$id <- ann_id; ann_id <- ann_id + 1
    coco$annotations[[length(coco$annotations) + 1]] <- ann
  }
  
  # Ajout des annotations pour les corners (catégorie 3)
  for (j in 1:nrow(corners)) {
    ann <- add_point_ann(img_id, 3, corners_px$px[j], corners_px$py[j], radius)
    ann$id <- ann_id; ann_id <- ann_id + 1
    coco$annotations[[length(coco$annotations) + 1]] <- ann
  }
  
  img_id <- img_id + 1
  
  # Mise à jour de la barre de progression
  if (i %% UPDATE_EVERY == 0 || i == N) {
    setTxtProgressBar(pb, i)
  }
  
}
close(pb)

write_json(
  coco,
  file.path(out_dir, "annotations.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

cat("Génération terminée:", length(coco$images), "images créées\n")