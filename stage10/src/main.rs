use anyhow::Result;
use clap::Parser;
use plotters::prelude::*;
use shared::point_explorer::PointExplorer;
use std::collections::HashMap;
use std::f64::consts::PI;
use uuid::Uuid;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    sim_map: String,
    #[arg(long, default_value = "similarity.png")]
    output: String,
    #[arg(long, default_value_t = 0.8)]
    threshold: f32,
    #[arg(long, default_value_t = 1200)]
    size: u32,
    #[arg(long, value_delimiter = ',')]
    #[arg(value_parser = clap::value_parser!(Uuid))]
    ids: Vec<Uuid>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let data = std::fs::read(r"img_sim_clean_new.bin")?;
    let sim_explorer: PointExplorer =
        bincode::serde::decode_from_slice(&data, bincode::config::standard())
            .expect("deserialize")
            .0;

    if args.ids.len() < 2 {
        eprintln!("need at least two ids");
        return Ok(());
    }

    let size = args.size;
    let center = (size as f64 / 2.0, size as f64 / 2.0);
    let radius = size as f64 * 0.4;

    let mut positions: HashMap<Uuid, (i32, i32)> = HashMap::new();
    for (i, id) in args.ids.iter().enumerate() {
        let angle = 2.0 * PI * i as f64 / args.ids.len() as f64;
        let x = (center.0 + radius * angle.cos()).round() as i32;
        let y = (center.1 + radius * angle.sin()).round() as i32;
        positions.insert(*id, (x, y));
    }

    let root = BitMapBackend::new(&args.output, (size, size)).into_drawing_area();
    root.fill(&WHITE)?;

    // draw edges
    for i in 0..args.ids.len() {
        for j in i + 1..args.ids.len() {
            let id1 = args.ids[i];
            let id2 = args.ids[j];
            let sim = sim_explorer.get_similarity((&id1, &id2))?;
            let (x1, y1) = positions[&id1];
            let (x2, y2) = positions[&id2];
            let low = sim < args.threshold;
            let color = if low { RED } else { BLUE };
            if low {
                println!("low similarity: {id1} <-> {id2}: {sim:.4}");
            }
            root.draw(&PathElement::new(
                vec![(x1, y1), (x2, y2)],
                color.stroke_width(2),
            ))?;
        }
    }

    // draw nodes
    for id in &args.ids {
        let (x, y) = positions[id];
        root.draw(&Circle::new((x, y), 5, BLACK.filled()))?;
        root.draw(&Text::new(
            id.to_string(),
            (x + 5, y + 5),
            ("sans-serif", 15).into_font(),
        ))?;
    }

    root.present()?;
    println!("saved visualization to {}", args.output);
    Ok(())
}
