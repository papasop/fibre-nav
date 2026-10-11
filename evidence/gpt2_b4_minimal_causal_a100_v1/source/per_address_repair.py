"""v1.9.5.2: training-only mixed-address constraint repair."""
import torch


ADDRESS_CONSTRAINT_MULTIPLIER = 4.0


def concept_prompts(node):
    description = node["description"]
    key = node["name"]
    suffix_a = f"[address={key}/amber]:"
    suffix_b = f"[address={key}/cedar]:"
    exact = [f"Concept memory {suffix_a}", f"Concept memory {suffix_b}"]
    prefixes = [
        "Retrieve the stored value from",
        f"For {description}, retrieve its stored value from",
        f"Look up the record described as {description} using",
        f"The requested field for {description} is stored at",
        f"Memory concerning {description} returns a value at",
        f"Answer the archive request for {description} from",
        f"CONCEPT={key}; CLUE={description}; READ",
        f"Disregard this weather note and recover {description} from",
        f"After reviewing the description {description}, consult",
        f"Verification request for {key} and {description}; location",
        f"A researcher requests the encoded value associated with {description} at",
        f"Short retrieval for {key} from",
        f"Database query: identify the saved symbol for {description} at",
        f"Given the clue {description}, obtain the corresponding entry from",
        f"Resolve this semantic record ({description}) by reading",
        f"The archive associates {description} with the value found at",
    ]
    train = []
    for prefix in prefixes:
        train.extend([f"{prefix} {suffix_a}", f"{prefix} {suffix_b}"])
    heldout_prefixes = [
        f"What symbol should be recovered for {description} from",
        f"Use this unseen wording to recall {description} at",
        f"Without repeating the training request, read the entry for {description} from",
        f"A new user asks for the saved value linked to {description}; consult",
    ]
    heldout = []
    for prefix in heldout_prefixes:
        heldout.extend([f"{prefix} {suffix_a}", f"{prefix} {suffix_b}"])
    return exact, train, heldout


def semantic_memory_loss(model, tok, prompts, targets, candidate_ids, device,
                         pair_count, pull_weight, consistency_weight,
                         worst_item_weight, worst_item_temperature,
                         margin_weight, margin_target,
                         slot_factorized=False, slot_contrast_weight=0.0,
                         slot_contrast_target=0.0):
    batch = tok(prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=48).to(device)
    out = model(**batch, output_hidden_states=True)
    last = batch["attention_mask"].sum(1) - 1
    rows = torch.arange(len(prompts), device=device)
    logits = out.logits[rows, last][:, candidate_ids]
    index = {token_id: i for i, token_id in enumerate(candidate_ids)}
    y = torch.tensor([index[t] for t in targets], device=device)
    per_item_ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    ce = per_item_ce.mean()
    hidden = out.hidden_states[-1][rows, last]
    if len(prompts) % pair_count:
        raise ValueError("Semantic prompts must contain complete view-major blocks")
    view_count = len(prompts) // pair_count
    tau = max(float(worst_item_temperature), 1e-6)

    if slot_factorized and pair_count == 2:
        ce_by_view = per_item_ce.reshape(view_count, pair_count)
        per_slot_ce = tau * torch.logsumexp(ce_by_view.T / tau, dim=1)
        # Retain the v1.9.5.1 smooth maximum across slots.
        robust_ce = tau * torch.logsumexp(per_slot_ce / tau, dim=0)
    else:
        robust_ce = tau * torch.logsumexp(per_item_ce / tau, dim=0)

    signed = (2 * y.float() - 1) * (logits[:, 1] - logits[:, 0])
    margin_penalty = torch.nn.functional.softplus(margin_target - signed)
    if slot_factorized and pair_count == 2:
        penalty_by_view = margin_penalty.reshape(view_count, pair_count)
        per_slot_margin = tau * torch.logsumexp(penalty_by_view.T / tau, dim=1)
        robust_margin = tau * torch.logsumexp(per_slot_margin / tau, dim=0)
        # v1.9.5.2 adds a conservative constraint only for mixed addresses
        # (01 or 10). Each slot must independently clear the declared training
        # margin. This uses training prompts only; held-out prompts are absent.
        signed_by_view = signed.reshape(view_count, pair_count)
        target_signs = (2 * y.float() - 1).reshape(view_count, pair_count)
        mixed_address = bool((target_signs[0, 0] != target_signs[0, 1]).item())
        if mixed_address:
            per_slot_soft_min = -tau * torch.logsumexp(
                -signed_by_view.T / tau, dim=1
            )
            address_constraint = (
                tau * torch.nn.functional.softplus(
                    (margin_target - per_slot_soft_min) / tau
                )
            ).sum()
            robust_margin = robust_margin + ADDRESS_CONSTRAINT_MULTIPLIER * address_constraint
    else:
        robust_margin = tau * torch.logsumexp(margin_penalty / tau, dim=0)

    h = torch.nn.functional.normalize(hidden, dim=-1).reshape(view_count, pair_count, -1)
    centroid = torch.nn.functional.normalize(h.mean(0), dim=-1)
    pull = (h - centroid.unsqueeze(0)).pow(2).sum(-1).mean()
    probabilities = torch.softmax(logits, dim=-1).reshape(view_count, pair_count, -1)
    mean_p = probabilities.mean(0).clamp_min(1e-8)
    consistency = (probabilities * (
        probabilities.clamp_min(1e-8).log() - mean_p.log().unsqueeze(0)
    )).sum(-1).mean()
    slot_contrast = logits.new_zeros(())
    if slot_factorized and pair_count == 2:
        delta = (logits[:, 1] - logits[:, 0]).reshape(view_count, pair_count)
        signs = (2 * y.float() - 1).reshape(view_count, pair_count)
        mixed = signs[:, 0] != signs[:, 1]
        if bool(mixed.any()):
            ordered_gap = ((signs[mixed, 1] - signs[mixed, 0])
                           * (delta[mixed, 1] - delta[mixed, 0]) / 2.0)
            slot_contrast = torch.nn.functional.softplus(
                slot_contrast_target - ordered_gap
            ).mean()
    total = (ce + worst_item_weight * robust_ce + margin_weight * robust_margin
             + pull_weight * pull + consistency_weight * consistency
             + slot_contrast_weight * slot_contrast)
    return total, ce, pull, consistency, robust_ce, robust_margin


def install(writer):
    writer.concept_prompts = concept_prompts
    writer.semantic_memory_loss = semantic_memory_loss
    return writer
