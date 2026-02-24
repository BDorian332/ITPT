# ==============================================================================
# Génération de dataset d'arbres phylogénétiques avec annotations COCO - Version 2
# ==============================================================================
#
# Nouveautés par rapport à la version 1:
#   - Génération de fichiers Newick (.nwk) pour chaque arbre
#   - Fichiers de métadonnées (CSV + JSONL) avec liens vers les Newick
#
# Structure COCO générée:
#   - Catégorie 1 = leaf (feuille)
#   - Catégorie 2 = internal_node (nœud interne)
#   - Catégorie 3 = corner (coin/virage)
#
# Sorties:
#   - images/*.png           : Images des arbres phylogénétiques
#   - newick/*.nwk          : Fichiers Newick (format texte des arbres)
#   - annotations.json      : Annotations COCO
#   - metadata.csv          : Métadonnées (ID, image, newick, nombre de feuilles, seed)
#   - metadata.jsonl        : Métadonnées au format JSONL
#
# Usage:
#   Rscript generate_dataset_V2.R
#   Rscript generate_dataset_V2.R --n=100 --width=1200 --height=1200
# ==============================================================================

suppressPackageStartupMessages({
  library(ape)
  library(jsonlite)
  library(png)
})

# PARSING DES ARGUMENTS EN LIGNE DE COMMANDE
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(key, default = NULL) {
  prefix <- paste0("--", key, "=")
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0) return(default)
  sub(prefix, "", hit[1], fixed = TRUE)
}

# CONFIG
# --- Chemins et répertoires ---
out_dir <- get_arg("out_dir", "./out_dataset")    # Dossier de sortie

# --- Dataset ---
N <- as.integer(get_arg("n", "10"))               # Nombre d'arbres à générer

# --- Dimensions des images ---
W <- as.integer(get_arg("width", "1000"))         # Largeur en pixels
H <- as.integer(get_arg("height", "1000"))        # Hauteur en pixels

# --- Nombre de feuilles par arbre ---
max_leaves <- as.integer(get_arg("leaves", "80")) # Maximum de feuilles
min_leaves <- as.integer(get_arg("min_leaves", "10"))  # Minimum de feuilles

# --- Épaisseur des branches ---
lwd_min <- as.numeric(get_arg("lwd_min", "1"))    # Épaisseur minimale
lwd_max <- as.numeric(get_arg("lwd_max", "5"))    # Épaisseur maximale

# --- Marges et padding ---
pad_frac <- as.numeric(get_arg("pad", "0.10"))    # Marge en fraction (0.10 = 10%)

# --- Annotations COCO ---
radius <- as.numeric(get_arg("radius", "6"))      # Rayon des points annotés (pixels)
k_poly <- as.integer(get_arg("k_poly", "16"))     # Nombre de points pour polygone (cercle)

# --- Labels des feuilles ---
label_len <- as.integer(get_arg("label_len", "6")) # Longueur des labels

# --- Espacement des labels ---
label_gap_min <- as.numeric(get_arg("label_gap_min", "0.01"))  # Gap minimal
label_gap_max <- as.numeric(get_arg("label_gap_max", "0.06"))  # Gap maximal
if (label_gap_min < 0 || label_gap_max < 0 || label_gap_min > label_gap_max) {
  stop("label_gap_min/max invalides")
}

# --- Graine aléatoire ---
seed <- get_arg("seed", NA)                       # Graine pour reproductibilité


if (!is.na(seed)) set.seed(as.integer(seed))
if (N <= 0) stop("n doit être > 0")
if (W <= 0 || H <= 0) stop("width/height doivent être > 0")
if (min_leaves < 2) min_leaves <- 2
if (max_leaves < min_leaves) stop("leaves (max) doit être >= min_leaves")
if (lwd_min <= 0 || lwd_max <= 0 || lwd_min > lwd_max) stop("lwd_min/lwd_max invalides")
if (pad_frac < 0) stop("pad doit être >= 0")
if (radius <= 0) stop("radius doit être > 0")

# Création des répertoires
img_dir <- file.path(out_dir, "images")
nwk_dir <- file.path(out_dir, "newick")
dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(nwk_dir, recursive = TRUE, showWarnings = FALSE)



rand_label_one <- function(len = 6) {
  #Génère un label aléatoire de longueur len avec lettres et chiffres.
  chars <- c(letters, LETTERS, 0:9)
  paste0(sample(chars, len, replace = TRUE), collapse = "")
}


unique_labels <- function(n, len = 6) {
  #Génère n labels uniques de longueur len.
  labs <- character(0)
  seen <- new.env(parent = emptyenv())
  while (length(labs) < n) {
    x <- rand_label_one(len)
    if (!exists(x, envir = seen, inherits = FALSE)) {
      assign(x, TRUE, envir = seen)
      labs <- c(labs, x)
    }
  }
  labs
}


write_lines <- function(path, text) {
  #Écrit du texte dans un fichier.
  con <- file(path, open = "wb")
  on.exit(close(con), add = TRUE)
  writeChar(text, con, eos = NULL)
}


compute_right_pad_x <- function(tr, cex, W, usr, plt, label_offset_x, extra_px = 10) {
  #Calcule le padding à droite nécessaire pour les labels.
  longest <- tr$tip.label[which.max(nchar(tr$tip.label))]
  w_px <- strwidth(longest, units = "inches", cex = cex) * par("dpi")
  plot_w_px <- (plt[2] - plt[1]) * W
  x_per_px <- (usr[2] - usr[1]) / plot_w_px
  label_offset_x + (w_px + extra_px) * x_per_px
}


make_ultrametric <- function(tr, internal_min = 0.05, internal_max = 1.2) {
  #  Transforme un arbre en arbre ultrametrique (toutes les feuilles alignées).  Les arbres ultramétriques sont typiques des arbres phylogénétiques.  
  nTips <- length(tr$tip.label)
  tr$edge.length <- runif(nrow(tr$edge), internal_min, internal_max)
  depths <- node.depth.edgelength(tr)
  max_tip <- max(depths[1:nTips])

  for (tip in 1:nTips) {
    idx <- which(tr$edge[, 2] == tip)
    if (length(idx) == 1) {
      delta <- max_tip - depths[tip]
      if (delta > 0) tr$edge.length[idx] <- tr$edge.length[idx] + delta
    }
  }
  tr
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
  #Crée une annotation COCO pour un point circulaire.
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

# Structure pour stocker les métadonnées de chaque arbre
meta <- data.frame(
  id = integer(0),
  image = character(0),
  newick_file = character(0),
  newick = character(0),
  leaves = integer(0),
  sample_seed = integer(0),
  stringsAsFactors = FALSE
)

# ==============================================================================
# BOUCLE PRINCIPALE - GÉNÉRATION DES ARBRES
# ==============================================================================
pb <- txtProgressBar(min = 0, max = N, style = 3)
UPDATE_EVERY <- 25

for (i in seq_len(N)) {
  sample_seed <- sample.int(.Machine$integer.max, 1)
  set.seed(sample_seed)

  n_leaves <- sample(min_leaves:max_leaves, 1)
  tr <- rtree(n_leaves)
  tr$tip.label <- unique_labels(n_leaves, label_len)
  tr <- make_ultrametric(tr)

  newick_str <- write.tree(tr)

  fname <- sprintf("tree_%06d.png", img_id)
  fpath <- file.path(img_dir, fname)
  nwk_name <- sprintf("tree_%06d.nwk", img_id)
  nwk_path <- file.path(nwk_dir, nwk_name)

  # Épaisseur aléatoire par arête
  edge_widths <- runif(nrow(tr$edge), min = lwd_min, max = lwd_max)

  max_depth <- max(node.depth.edgelength(tr)[1:length(tr$tip.label)])
  pad_x <- pad_frac * max_depth
  pad_y <- max(0.5, (pad_frac/2) * n_leaves)

  label_gap <- runif(1, label_gap_min, label_gap_max)

  # Rendu de l'arbre
  png(fpath, width = W, height = H, bg = "white")
  par(mar = c(0, 0, 0, 0), xaxs = "i", yaxs = "i")
  plot.phylo(
    tr,
    type = "phylogram",
    direction = "rightwards",
    show.tip.label = TRUE,
    cex = 1.5, 
    label.offset = label_gap,
    no.margin = TRUE,
    edge.width = edge_widths,
    x.lim = c(-pad_x, max_depth + pad_x),
    y.lim = c(1 - pad_y, n_leaves + pad_y)
  )

  lp <- get("last_plot.phylo", envir = .PlotPhyloEnv)
  xx <- lp$xx
  yy <- lp$yy
  usr <- par("usr")
  plt <- par("plt")
  dev.off()

  add_right_px <- 100
  img <- readPNG(fpath)
  h <- dim(img)[1]; w <- dim(img)[2]
  channels <- dim(img)[3]

  new_img <- array(1, dim = c(h, w + add_right_px, channels))
  new_img[, 1:w, ] <- img
  writePNG(new_img, fpath)

  # Sauvegarde du fichier Newick
  write_lines(nwk_path, paste0(newick_str, "\n"))

  # ANNOTATIONS COCO

  nTip <- length(tr$tip.label)
  tips <- 1:nTip
  nodes <- (nTip + 1):(nTip + tr$Nnode)

  # Extraction des corners (extrémités des verticales)
  parents <- tr$edge[, 1]
  children <- tr$edge[, 2]
  internal_parents <- unique(parents[parents > nTip])

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
  corners <- if (length(corner_list) > 0) do.call(rbind, corner_list) else matrix(numeric(0), ncol = 2)
  if (nrow(corners) > 0) corners <- unique(corners)

  # Conversion des coordonnées en pixels
  tips_px <- to_px(xx[tips], yy[tips], usr, W, H, plt)
  nodes_px <- to_px(xx[nodes], yy[nodes], usr, W, H, plt)

  if (nrow(corners) > 0) {
    corners_px <- to_px(corners[, 1], corners[, 2], usr, W, H, plt)
  } else {
    corners_px <- list(px = numeric(0), py = numeric(0))
  }

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
  if (length(corners_px$px) > 0) {
    for (j in seq_along(corners_px$px)) {
      ann <- add_point_ann(img_id, 3, corners_px$px[j], corners_px$py[j], radius)
      ann$id <- ann_id; ann_id <- ann_id + 1
      coco$annotations[[length(coco$annotations) + 1]] <- ann
    }
  }

  meta <- rbind(meta, data.frame(
    id = img_id,
    image = file.path("images", fname),
    newick_file = file.path("newick", nwk_name),
    newick = newick_str,
    leaves = n_leaves,
    sample_seed = sample_seed,
    stringsAsFactors = FALSE
  ))

  img_id <- img_id + 1
  
  # Mise à jour de la barre de progression
  if (i %% UPDATE_EVERY == 0 || i == N) setTxtProgressBar(pb, i)
}
close(pb)

# Sauvegarde des annotations COCO
write_json(
  coco,
  file.path(out_dir, "annotations.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

# Sauvegarde des métadonnées au format CSV
write.csv(meta, file.path(out_dir, "metadata.csv"), row.names = FALSE)

# Sauvegarde des métadonnées au format JSONL (une ligne JSON par arbre)
jsonl_path <- file.path(out_dir, "metadata.jsonl")
con <- file(jsonl_path, open = "wb")
on.exit(close(con), add = TRUE)

esc <- function(s) gsub('(["\\\\])', '\\\\\\1', s)

for (k in seq_len(nrow(meta))) {
  line <- paste0(
    "{",
    "\"id\":", meta$id[k], ",",
    "\"image\":\"", esc(meta$image[k]), "\",",
    "\"newick_file\":\"", esc(meta$newick_file[k]), "\",",
    "\"newick\":\"", esc(meta$newick[k]), "\",",
    "\"leaves\":", meta$leaves[k], ",",
    "\"sample_seed\":", meta$sample_seed[k],
    "}\n"
  )
  writeChar(line, con, eos = NULL)
}

cat("Génération terminée:", length(coco$images), "images |", length(coco$annotations), "annotations\n")
cat("Dossier de sortie:", normalizePath(out_dir), "\n")