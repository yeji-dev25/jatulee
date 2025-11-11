package com.p_project.writing;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class WritingSessionService {

    private final WritingSessionRepository writingSessionRepository;

    public List<WritingSessionDTO> getRecentWritingSessions(Long userId) {
        Pageable limitFive = PageRequest.of(0, 5);
        return writingSessionRepository.findRecentWritingSessions(userId, limitFive)
                .stream()
                .map(w -> new WritingSessionDTO(
                        w.getTitle(),
                        w.getType().name(),
                        w.getGenre(),
                        w.getEmotion(),
                        w.getCreatedAt()
                ))
                .collect(Collectors.toList());
    }
}
