package com.p_project.writing;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface WritingSessionRepository extends JpaRepository<WritingSessionEntity, Long> {

    @Query("SELECT w FROM WritingSessionEntity w " +
            "WHERE w.userId = :userId AND w.deletedAt IS NULL " +
            "AND (w.type = 'diary' OR w.type = 'book') " +
            "ORDER BY w.createdAt DESC")
    List<WritingSessionEntity> findRecentWritingSessions(Long userId, Pageable pageable);
}
